"""SetFit Stage 2: Train MLP classification head on cached embeddings.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys
import time
import logging
import argparse
import sqlite3
import joblib

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from db import get_reader_conn
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_reader_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)



def get_active_tag_map(db_path: str) -> dict[int, int]:
    """Returns {db_tag_id: contiguous_idx} for tags actually used in cleaned_document_tags.
    Strips unused tags from the vocabulary to avoid dead output neurons."""
    conn = get_reader_conn(db_path)
    rows = conn.execute(
        "SELECT DISTINCT tag_id FROM cleaned_document_tags ORDER BY tag_id"
    ).fetchall()
    conn.close()
    return {tid: i for i, (tid,) in enumerate(rows)}


def load_split(db_path: str, split: str, tag_map: dict[int, int]) -> tuple[list[int], csr_matrix, int]:
    """Build sparse label matrix directly from (chunk_id, tag_id) pairs.
    Skips SQL GROUP BY (which uses a 55M-row temp B-tree); aggregates in Python."""
    t0 = time.time()
    num_tags = len(tag_map)
    conn = get_reader_conn(db_path)

    # Stream (chunk_id, tag_id) pairs ordered by chunk_id.
    pairs = conn.execute(
        """
        SELECT s.chunk_id, dt.tag_id
        FROM chunk_train_samples s
        JOIN chunks c ON s.chunk_id = c.chunk_id
        JOIN cleaned_document_tags dt ON c.doc_id = dt.doc_id
        WHERE s.split = ?
        ORDER BY s.chunk_id
        """,
        (split,),
    ).fetchall()
    conn.close()
    logger.info(f"  [{split}] {len(pairs):,} (chunk,tag) pairs fetched in {time.time()-t0:.1f}s")

    t1 = time.time()
    chunk_ids: list[int] = []
    chunk_to_row: dict[int, int] = {}
    row_idx = np.empty(len(pairs), dtype=np.int64)
    col_idx = np.empty(len(pairs), dtype=np.int32)
    out_pos = 0
    for cid, tid in pairs:
        r = chunk_to_row.get(cid)
        if r is None:
            r = len(chunk_ids)
            chunk_to_row[cid] = r
            chunk_ids.append(cid)
        idx = tag_map.get(tid)
        if idx is None:
            continue
        row_idx[out_pos] = r
        col_idx[out_pos] = idx
        out_pos += 1

    row_idx = row_idx[:out_pos]
    col_idx = col_idx[:out_pos]
    data = np.ones(out_pos, dtype=np.float32)
    n = len(chunk_ids)
    Y = csr_matrix((data, (row_idx, col_idx)), shape=(n, num_tags))
    logger.info(f"  [{split}] {n:,} samples, {num_tags} tags, "
                f"{Y.nnz:,} nonzero labels — built in {time.time()-t0:.1f}s "
                f"(aggregation {time.time()-t1:.1f}s)")
    return chunk_ids, Y, num_tags


def fetch_chunk_texts(db_path: str, chunk_ids: list[int]) -> list[str]:
    """Returns texts in the SAME ORDER as chunk_ids."""
    conn = get_reader_conn(db_path)
    conn.execute("CREATE TEMP TABLE _cids (chunk_id INTEGER PRIMARY KEY)")
    conn.executemany("INSERT INTO _cids VALUES (?)", [(c,) for c in chunk_ids])
    text_map = dict(
        conn.execute(
            "SELECT c.chunk_id, c.chunk_text FROM chunks c JOIN _cids t ON c.chunk_id = t.chunk_id"
        ).fetchall()
    )
    conn.close()
    return [text_map.get(cid, "") for cid in chunk_ids]



def build_embedding_cache(
    db_path: str,
    encoder_path: str,
    cache_dir: str,
    encode_batch_size: int = 512,
    tag_map: dict[int, int] = None,
):
    """Encodes train + val splits to fp16 memmap files."""
    os.makedirs(cache_dir, exist_ok=True)
    if tag_map is None:
        tag_map = get_active_tag_map(db_path)

    probe = SentenceTransformer(encoder_path).to("cuda:0")
    dim = probe.get_embedding_dimension()
    del probe
    torch.cuda.empty_cache()

    for split in ("train", "val"):
        emb_path = os.path.join(cache_dir, f"{split}_emb.f16.npy")
        ids_path = os.path.join(cache_dir, f"{split}_ids.npy")
        y_path   = os.path.join(cache_dir, f"{split}_labels.npz")

        if os.path.exists(emb_path) and os.path.exists(ids_path) and os.path.exists(y_path):
            logger.info(f"[{split}] cache exists, skipping encode")
            continue

        logger.info(f"[{split}] loading labels...")
        chunk_ids, Y, _ = load_split(db_path, split, tag_map)
        logger.info(f"[{split}] fetching texts...")
        texts = fetch_chunk_texts(db_path, chunk_ids)

        n = len(texts)
        emb = np.memmap(emb_path, dtype=np.float16, mode="w+", shape=(n, dim))

        world_size = torch.cuda.device_count()
        logger.info(f"[{split}] encoding {n:,} chunks on {world_size} GPU(s) "
                    f"(batch={encode_batch_size}, fp16)...")

        if world_size > 1:
            # Write texts to a binary file with an offset index
            texts_path = os.path.join(cache_dir, f"{split}_texts.bin")
            offsets_path = os.path.join(cache_dir, f"{split}_texts.off.npy")
            offsets = np.empty(n + 1, dtype=np.int64)
            offsets[0] = 0
            with open(texts_path, "wb") as fh:
                for i, t in enumerate(texts):
                    b = t.encode("utf-8")
                    fh.write(b)
                    offsets[i + 1] = offsets[i] + len(b)
            np.save(offsets_path, offsets)
            del texts

            emb.flush()
            del emb

            mp.spawn(
                _encode_worker,
                args=(world_size, encoder_path, texts_path, offsets_path,
                      emb_path, n, dim, encode_batch_size),
                nprocs=world_size,
                join=True,
            )
            os.remove(texts_path)
            os.remove(offsets_path)
        else:
            _encode_single(encoder_path, texts, emb, encode_batch_size, device="cuda:0")
            emb.flush()
            del emb

        np.save(ids_path, np.array(chunk_ids, dtype=np.int64))
        save_npz(y_path, Y)

    # Persist tag_map so inference code can round-trip contiguous idx -> db tag_id
    import json
    with open(os.path.join(cache_dir, "tag_map.json"), "w") as f:
        json.dump({str(k): v for k, v in tag_map.items()}, f)
    logger.info(f"Embedding cache written → {cache_dir}")


def _encode_single(encoder_path, texts, emb_out, batch_size, device):
    encoder = SentenceTransformer(encoder_path).to(device)
    encoder.eval()
    encoder.half()  # fp16 for 2x throughput on 3090

    n = len(texts)
    with torch.inference_mode():
        for start in tqdm(range(0, n, batch_size), desc=f"Encoding ({device})",
                          dynamic_ncols=True):
            end = min(start + batch_size, n)
            vecs = encoder.encode(
                texts[start:end],
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            emb_out[start:end] = vecs.astype(np.float16)


def _encode_worker(rank, world_size, encoder_path, texts_path, offsets_path,
                   emb_path, n, dim, batch_size):
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    per_rank = (n + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, n)

    offsets = np.load(offsets_path, mmap_mode="r")
    import mmap
    text_fh = open(texts_path, "rb")
    text_mm = mmap.mmap(text_fh.fileno(), 0, prot=mmap.PROT_READ)

    def get_text(global_idx):
        o0 = int(offsets[global_idx])
        o1 = int(offsets[global_idx + 1])
        return text_mm[o0:o1].decode("utf-8")

    my_texts = [get_text(i) for i in range(start, end)]
    text_mm.close()
    text_fh.close()
    del offsets

    emb = np.memmap(emb_path, dtype=np.float16, mode="r+", shape=(n, dim))

    encoder = SentenceTransformer(encoder_path).to(device)
    encoder.eval()
    encoder.half()
    tokenizer = encoder.tokenizer

    # one thread tokenizes the next batch while GPU encodes the current one
    from concurrent.futures import ThreadPoolExecutor

    def tokenize_batch(batch_texts):
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,  # most chunks are <256 tokens; saves padding waste
            return_tensors="pt",
        )

    n_local = len(my_texts)
    indices = list(range(0, n_local, batch_size))

    with torch.inference_mode(), ThreadPoolExecutor(max_workers=2) as pool:
        next_future = pool.submit(tokenize_batch, my_texts[0:min(batch_size, n_local)])

        pbar = tqdm(indices, desc=f"Encoding rank{rank}", disable=(rank != 0),
                    dynamic_ncols=True)
        for i, local_start in enumerate(pbar):
            local_end = min(local_start + batch_size, n_local)

            enc = next_future.result()

            if i + 1 < len(indices):
                nxt_start = indices[i + 1]
                nxt_end = min(nxt_start + batch_size, n_local)
                next_future = pool.submit(tokenize_batch, my_texts[nxt_start:nxt_end])

            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            out = encoder(enc)
            vecs = out["sentence_embedding"].cpu().numpy()
            emb[start + local_start : start + local_end] = vecs.astype(np.float16)

    emb.flush()
    del emb


# Dataset
class CachedEmbeddingDataset(Dataset):
    """Returns just the row index; the collate_fn does batched memmap reads
    and sparse→dense label conversion in one shot (much faster than per-sample)."""
    def __init__(self, emb_path: str, n: int, dim: int, Y: csr_matrix):
        self.emb_path = emb_path
        self.dim = dim
        self.n = n
        self.Y = Y
        self._emb = None

    @property
    def emb(self):
        if self._emb is None:
            self._emb = np.memmap(self.emb_path, dtype=np.float16, mode="r",
                                  shape=(self.n, self.dim))
        return self._emb

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return int(idx)


class BatchedCollate:
    """Module-level callable so it's picklable under forkserver/spawn."""
    def __init__(self, emb_path: str, n: int, dim: int, Y: csr_matrix):
        self.emb_path = emb_path
        self.n = n
        self.dim = dim
        self.Y = Y
        self._emb = None

    @property
    def emb(self):
        if self._emb is None:
            self._emb = np.memmap(self.emb_path, dtype=np.float16, mode="r",
                                  shape=(self.n, self.dim))
        return self._emb

    def __call__(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        x = torch.from_numpy(self.emb[idx].astype(np.float32, copy=False))
        y = torch.from_numpy(self.Y[idx].toarray().astype(np.float32, copy=False))
        return x, y


# Model
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels),
        )

    def forward(self, x):
        return self.net(x)


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "23568"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def train_worker(rank, world_size, cache_dir, dim, num_tags,
                 n_train, n_val, output_path, epochs, batch_size, lr):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    Y_train = load_npz(os.path.join(cache_dir, "train_labels.npz"))

    train_ds = CachedEmbeddingDataset(
        os.path.join(cache_dir, "train_emb.f16.npy"), n_train, dim, Y_train
    )
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True)
    train_collate = BatchedCollate(os.path.join(cache_dir, "train_emb.f16.npy"),
                                   n_train, dim, Y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              collate_fn=train_collate,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)

    # Val data only needed on rank 0
    if rank == 0:
        Y_val = load_npz(os.path.join(cache_dir, "val_labels.npz"))
        val_ds = CachedEmbeddingDataset(
            os.path.join(cache_dir, "val_emb.f16.npy"), n_val, dim, Y_val
        )
        val_collate = BatchedCollate(os.path.join(cache_dir, "val_emb.f16.npy"),
                                     n_val, dim, Y_val)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=val_collate,
                                num_workers=2, pin_memory=True,
                                persistent_workers=True, prefetch_factor=4)

    classifier = MultiLabelClassifier(dim, num_tags).to(device)
    ddp = DDP(classifier, device_ids=[rank], output_device=rank)

    # No pos_weight
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(ddp.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")

    if rank == 0:
        logger.info(f"Training MLP: {n_train:,} train / {n_val:,} val | "
                    f"{num_tags} tags | {epochs} epochs | {world_size} GPUs (train) "
                    f"| 1 GPU (val)")

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        ddp.train()
        total_loss = 0.0
        f1_thresh = 0.375
        tp = fp = fn = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]",
                    disable=(rank != 0), dynamic_ncols=True)
        for step, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = ddp(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

            with torch.no_grad():
                pred = (torch.sigmoid(logits.detach().float()) >= f1_thresh)
                y_bool = y.bool()
                tp += int((pred & y_bool).sum().item())
                fp += int((pred & ~y_bool).sum().item())
                fn += int((~pred & y_bool).sum().item())

            if rank == 0:
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec  = tp / (tp + fn) if (tp + fn) else 0.0
                mf1  = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}",
                                 mF1=f"{mf1:.3f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}",
                                 refresh=False)

        avg_train = total_loss / len(train_loader)

        if rank == 0:
            classifier.eval()
            val_loss = 0.0
            # Accumulate TP/FP/FN
            thresholds = [0.15, 0.25, 0.375, 0.50]
            counters = {t: [0, 0, 0] for t in thresholds}  # [tp, fp, fn]
            with torch.no_grad():
                for x, y in tqdm(val_loader,
                                 desc=f"Epoch {epoch+1}/{epochs} [Val]",
                                 dynamic_ncols=True):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = classifier(x)
                        val_loss += criterion(logits, y).item()
                    probs = torch.sigmoid(logits.float())
                    y_bool = y.bool()
                    for t in thresholds:
                        pred = probs >= t
                        counters[t][0] += int((pred & y_bool).sum().item())
                        counters[t][1] += int((pred & ~y_bool).sum().item())
                        counters[t][2] += int((~pred & y_bool).sum().item())

            avg_val = val_loss / len(val_loader)

            # Log F1 at each threshold
            f1s = {}
            for t in thresholds:
                tp, fp, fn = counters[t]
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                f1s[t] = 2.0 * p * r / (p + r) if (p + r) else 0.0

            f1_str = "  ".join(f"t={t}:{f1s[t]:.4f}" for t in thresholds)
            logger.info(f"Epoch {epoch+1}: train_loss={avg_train:.4f}  "
                        f"val_loss={avg_val:.4f}  {f1_str}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    "state_dict": classifier.state_dict(),
                    "val_loss": best_val_loss,
                    "epoch": epoch + 1,
                }, output_path)
                logger.info(f"  → new best val_loss={best_val_loss:.4f}, saved")

        dist.barrier()

    if rank == 0:
        logger.info(f"Training done. Best val_loss={best_val_loss:.4f}")

    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",          default="nlp.db")
    parser.add_argument("--encoder",     required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--cache-dir",   default="models/final_deep/stage2_cache")
    parser.add_argument("--epochs",      type=int,   default=40)
    parser.add_argument("--batch-size",  type=int,   default=4096,
                        help="MLP train batch (can be much larger than encoder batch)")
    parser.add_argument("--encode-batch",type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=2e-3)
    parser.add_argument("--skip-encode", action="store_true",
                        help="Use existing cache, skip encoding pass")
    args = parser.parse_args()

    mp.set_start_method("fork", force=True)

    # contiguous 0..N-1 indices for tags actually in use
    tag_map = get_active_tag_map(args.db)
    num_tags = len(tag_map)
    logger.info(f"Active tag vocabulary: {num_tags} (from {sum(1 for _ in tag_map):,} tags)")

    # encode once (both GPUs)
    if not args.skip_encode:
        build_embedding_cache(args.db, args.encoder, args.cache_dir,
                              args.encode_batch, tag_map=tag_map)

    probe = SentenceTransformer(args.encoder).to("cuda:0")
    dim = probe.get_embedding_dimension()
    del probe
    torch.cuda.empty_cache()

    n_train = int(np.load(os.path.join(args.cache_dir, "train_ids.npy")).shape[0])
    n_val   = int(np.load(os.path.join(args.cache_dir, "val_ids.npy")).shape[0])

    # DDP train (both GPUs), val on rank 0 only
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_worker,
        args=(world_size, args.cache_dir, dim, num_tags,
              n_train, n_val, args.output, args.epochs, args.batch_size, args.lr),
        nprocs=world_size,
        join=True,
    )

    # save joblib metadata bundle
    conn = get_reader_conn(args.db)
    tag_name_map = dict(conn.execute("SELECT tag_id, tag_name FROM cleaned_tags").fetchall())
    conn.close()

    idx_to_tagid = [None] * num_tags
    for db_tid, idx in tag_map.items():
        idx_to_tagid[idx] = db_tid
    tag_ids   = idx_to_tagid
    tag_names = [tag_name_map.get(tid, f"<tag:{tid}>") for tid in tag_ids]

    ckpt = torch.load(args.output, map_location="cpu", weights_only=False)
    joblib_path = args.output.replace(".pth", ".joblib")
    joblib.dump({
        "model_state":   ckpt["state_dict"],
        "best_val_loss": ckpt.get("val_loss", None),
        "input_dim":     dim,
        "num_labels":    num_tags,
        "tag_names":     tag_names,
        "tag_ids":       tag_ids,
        "tag_map":       tag_map,
        "encoder_path":  args.encoder,
    }, joblib_path)
    logger.info(f"Saved joblib → {joblib_path}")


if __name__ == "__main__":
    main()
