"""(Optional) Fine-tune the FLAN-T5 grounded generative reader.

Trains on (question + context → answer) from ``neural-bridge/rag-dataset-12000``
with a strict grounded prompt and an "I don't know" instruction. **bf16 only**
(FLAN-T5 produces NaNs under fp16). Resume-safe. Saves to ``models/generator/latest``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..models.model_registry import save_model_metadata
from ..models.reader_generative import PROMPT

logger = get_logger(__name__)


def _first_col(ds, cands):
    for c in cands:
        if c in ds.column_names:
            return c
    return None


def train_generator(cfg: AppConfig, limit: Optional[int] = None) -> Dict:
    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, get_last_checkpoint)

    gcfg = cfg.generator
    raw = load_dataset(cfg.data.generative_rag_dataset)
    train = raw["train"]
    if limit:
        train = train.select(range(min(limit, len(train))))
    q_col = _first_col(train, ["question", "query"])
    c_col = _first_col(train, ["context", "passage", "document"])
    a_col = _first_col(train, ["answer", "answers", "response"])

    tokenizer = AutoTokenizer.from_pretrained(gcfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(gcfg.model_name)

    def _prep(ex):
        inputs = [PROMPT.format(context=c, question=q) for q, c in zip(ex[q_col], ex[c_col])]
        model_in = tokenizer(inputs, max_length=gcfg.max_input_length, truncation=True)
        labels = tokenizer(text_target=[str(a) for a in ex[a_col]], max_length=gcfg.max_target_length, truncation=True)
        model_in["labels"] = labels["input_ids"]
        return model_in

    train_tok = train.map(_prep, batched=True, remove_columns=train.column_names)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    sig = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
    kw = dict(output_dir=str(Path(gcfg.output_dir) / "_ckpt"), num_train_epochs=gcfg.num_train_epochs,
              learning_rate=gcfg.learning_rate, per_device_train_batch_size=gcfg.per_device_train_batch_size,
              gradient_accumulation_steps=gcfg.gradient_accumulation_steps, weight_decay=0.01,
              label_smoothing_factor=gcfg.label_smoothing_factor, bf16=bf16, fp16=False,  # never fp16 for FLAN-T5
              predict_with_generate=True, save_strategy="epoch", save_total_limit=2,
              logging_steps=50, report_to="none")
    args = Seq2SeqTrainingArguments(**{k: v for k, v in kw.items() if k in sig})

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_tok, data_collator=collator, tokenizer=tokenizer)
    ckpt = Path(gcfg.output_dir) / "_ckpt"
    last = get_last_checkpoint(str(ckpt)) if any(ckpt.glob("checkpoint-*")) else None
    logger.info("Training FLAN-T5 generator %s (bf16=%s) | %d examples", gcfg.model_name, bf16, len(train_tok))
    trainer.train(resume_from_checkpoint=last)

    final = Path(gcfg.output_dir) / "latest"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    save_model_metadata(final, base_model=gcfg.model_name, task="generative-reader",
                        config_subset={"epochs": gcfg.num_train_epochs, "lr": gcfg.learning_rate},
                        dataset_info={"dataset": cfg.data.generative_rag_dataset, "train": len(train)})
    logger.info("Generator saved -> %s", final)
    return {"model_dir": str(final), "n_examples": len(train_tok)}


__all__ = ["train_generator"]
