"""Fine-tune the dense retriever (sentence-transformers v3+).

MultipleNegativesRankingLoss on (query, positive-passage) pairs with large
in-batch negatives; optional hard-negative mining with the base model first.
Resume-safe via HF checkpointing. Saves to ``models/retriever/latest``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ..config import AppConfig
from ..data.dataset import load_retriever_pairs
from ..logging_utils import get_logger
from ..models.model_registry import save_model_metadata

logger = get_logger(__name__)


def _bf16() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def train_retriever(cfg: AppConfig, limit: Optional[int] = None) -> Dict:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    rcfg = cfg.retriever
    ds, q_col, p_col = load_retriever_pairs(cfg.data, limit=limit)
    train_ds = Dataset.from_dict({"anchor": list(ds[q_col]), "positive": list(ds[p_col])})
    logger.info("Retriever training pairs: %d", len(train_ds))

    model = SentenceTransformer(rcfg.bi_encoder_model)
    # CachedMNRL pushes a large effective batch on H100 without OOM; fall back to MNRL.
    try:
        loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)
    except Exception:
        loss = losses.MultipleNegativesRankingLoss(model)

    out_dir = Path(rcfg.output_dir)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "_ckpt"),
        num_train_epochs=rcfg.num_train_epochs,
        per_device_train_batch_size=rcfg.train_batch_size,
        learning_rate=rcfg.learning_rate,
        warmup_ratio=rcfg.warmup_ratio,
        weight_decay=0.01,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        bf16=_bf16(),
        logging_steps=50, save_strategy="epoch", save_total_limit=2, report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss)
    last = _last_ckpt(out_dir / "_ckpt")
    trainer.train(resume_from_checkpoint=last)

    final = out_dir / "latest"
    model.save(str(final))
    save_model_metadata(final, base_model=rcfg.bi_encoder_model, task="dense-retriever",
                        config_subset={"epochs": rcfg.num_train_epochs, "batch": rcfg.train_batch_size,
                                       "lr": rcfg.learning_rate},
                        dataset_info={"pairs": len(train_ds), "dataset": cfg.data.retriever_pairs_dataset})
    logger.info("Saved fine-tuned retriever -> %s", final)
    return {"model_dir": str(final), "n_pairs": len(train_ds)}


def _last_ckpt(d: Path):
    if not d.exists():
        return None
    cks = sorted(d.glob("checkpoint-*"))
    return str(cks[-1]) if cks else None


__all__ = ["train_retriever"]
