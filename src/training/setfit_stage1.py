import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import time

try:
    from db import get_reader_conn
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_reader_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_pairs_and_texts(db_path: str) -> tuple[list[str], list[str], list[float]]:
    t0 = time.time()
    conn = get_reader_conn(db_path)

    logger.info("Loading contrastive pairs...")
    pairs = conn.execute(
        "SELECT sample_a_id, sample_b_id, similarity FROM contrastive_pairs"
    ).fetchall()
    logger.info(f"  {len(pairs):,} pairs in {time.time()-t0:.1f}s")

    needed_ids = set()
    for a, b, _ in pairs:
        needed_ids.add(a)
        needed_ids.add(b)
    logger.info(f"  {len(needed_ids):,} unique chunk IDs")

    t1 = time.time()
    conn.execute("CREATE TEMP TABLE _needed (chunk_id INTEGER PRIMARY KEY)")
    conn.executemany("INSERT INTO _needed VALUES (?)", [(i,) for i in needed_ids])
    chunk_text: dict[int, str] = dict(
        conn.execute(
            "SELECT c.chunk_id, c.chunk_text "
            "FROM chunks c JOIN _needed n ON c.chunk_id = n.chunk_id"
        ).fetchall()
    )
    conn.close()
    logger.info(f"  Chunk texts fetched in {time.time()-t1:.1f}s")

    texts_a, texts_b, sims = [], [], []
    missing = 0
    for a, b, sim in pairs:
        ta, tb = chunk_text.get(a), chunk_text.get(b)
        if ta is None or tb is None:
            missing += 1
            continue
        texts_a.append(ta)
        texts_b.append(tb)
        sims.append(float(sim))

    if missing:
        logger.warning(f"  Skipped {missing:,} pairs with missing chunk text")
    logger.info(f"  {len(texts_a):,} usable pairs assembled in {time.time()-t0:.1f}s total")
    return texts_a, texts_b, sims


class ContrastiveDataset(Dataset):
    def __init__(self, texts_a, texts_b, sims):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.sims = sims

    def __len__(self):
        return len(self.sims)

    def __getitem__(self, idx):
        return InputExample(
            texts=[self.texts_a[idx], self.texts_b[idx]],
            label=self.sims[idx],
        )


def move_features_to_device(features, device):
    """Move only tensor values in feature dicts — skip strings and other non-tensors."""
    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in f.items()}
        for f in features
    ]


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "23567"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_stage1(rank, world_size, texts_a, texts_b, sims, model_name, output_path, epochs, batch_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model = SentenceTransformer(model_name)
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    train_loss = losses.CosineSimilarityLoss(model)

    dataset = ContrastiveDataset(texts_a, texts_b, sims)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=model.smart_batching_collate,
        num_workers=0,
        pin_memory=True,
    )

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * 0.1)

    optimizer = AdamW(ddp_model.parameters(), lr=2e-5, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    if rank == 0:
        logger.info(f"Rank 0: {steps_per_epoch} steps/epoch x {epochs} epochs, {warmup_steps} warmup")

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            disable=(rank != 0),
            dynamic_ncols=True,
        )

        for step, batch in enumerate(pbar):
            features, labels = batch
            features = move_features_to_device(features, device)
            labels = labels.to(device)

            loss = train_loss(features, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            avg = total_loss / (step + 1)
            pbar.set_postfix(loss=f"{avg:.4f}", refresh=False)

        if rank == 0:
            logger.info(f"Epoch {epoch+1} complete — avg loss: {total_loss/steps_per_epoch:.4f}")

    if rank == 0:
        model.save(output_path)
        logger.info(f"Model saved to {output_path}")

    cleanup()


if __name__ == "__main__":
    import argparse

    mp.set_start_method("fork", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    texts_a, texts_b, sims = load_pairs_and_texts(args.db)

    world_size = torch.cuda.device_count()
    logger.info(f"Stage 1: {len(texts_a):,} pairs across {world_size} GPU(s)")

    mp.spawn(
        train_stage1,
        args=(world_size, texts_a, texts_b, sims, args.model, args.output, args.epochs, args.batch_size),
        nprocs=world_size,
        join=True,
    )
