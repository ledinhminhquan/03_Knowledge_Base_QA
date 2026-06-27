"""Fine-tune the extractive reader on SQuAD v2 (doc_stride + no-answer).

Implements the canonical HF question-answering preprocessing: sliding-window
features (``doc_stride``), answer-span → token positions, and **no-answer ⇒
point to [CLS]** (the abstention signal). After training we score EM/F1 (incl.
HasAns/NoAns) on a validation subset via the QA pipeline. Resume-safe.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Optional

from ..config import AppConfig
from ..data.dataset import load_squad
from ..data.preprocessing import normalize_answer
from ..logging_utils import get_logger
from ..models.model_registry import save_model_metadata

logger = get_logger(__name__)


def _prepare_train_features(examples, tokenizer, max_length, doc_stride):
    tok = tokenizer(examples["question"], examples["context"], truncation="only_second",
                    max_length=max_length, stride=doc_stride, return_overflowing_tokens=True,
                    return_offsets_mapping=True, padding="max_length")
    sample_map = tok.pop("overflow_to_sample_mapping")
    offsets_all = tok.pop("offset_mapping")
    tok["start_positions"], tok["end_positions"] = [], []
    for i, offsets in enumerate(offsets_all):
        input_ids = tok["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0
        seq_ids = tok.sequence_ids(i)
        answers = examples["answers"][sample_map[i]]
        if len(answers["answer_start"]) == 0:
            tok["start_positions"].append(cls_index); tok["end_positions"].append(cls_index); continue
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        ts = 0
        while ts < len(seq_ids) and seq_ids[ts] != 1:
            ts += 1
        te = len(input_ids) - 1
        while te >= 0 and seq_ids[te] != 1:
            te -= 1
        if not (offsets[ts][0] <= start_char and offsets[te][1] >= end_char):
            tok["start_positions"].append(cls_index); tok["end_positions"].append(cls_index)
        else:
            while ts < len(offsets) and offsets[ts][0] <= start_char:
                ts += 1
            tok["start_positions"].append(ts - 1)
            while offsets[te][1] >= end_char:
                te -= 1
            tok["end_positions"].append(te + 1)
    return tok


def _build_args(rcfg, output_dir, bf16, fp16):
    from transformers import TrainingArguments
    sig = set(inspect.signature(TrainingArguments.__init__).parameters)
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    kw = dict(output_dir=str(output_dir), num_train_epochs=rcfg.num_train_epochs,
              learning_rate=rcfg.learning_rate, per_device_train_batch_size=rcfg.per_device_train_batch_size,
              per_device_eval_batch_size=rcfg.per_device_train_batch_size * 2, warmup_ratio=rcfg.warmup_ratio,
              weight_decay=rcfg.weight_decay, save_strategy="epoch", save_total_limit=2,
              logging_steps=50, report_to="none", seed=rcfg.seed, bf16=bf16, fp16=fp16)
    kw[eval_key] = "epoch"
    return TrainingArguments(**{k: v for k, v in kw.items() if k in sig})


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = {}
    for w in p:
        if w in g:
            common[w] = common.get(w, 0) + 1
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec, rec = num_same / len(p), num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def train_reader(cfg: AppConfig, limit: Optional[int] = None, eval_n: int = 500) -> Dict:
    import torch
    from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                              DataCollatorWithPadding, default_data_collator, get_last_checkpoint, Trainer)

    rcfg = cfg.reader
    raw = load_squad(cfg.data)
    train = raw["train"]
    val = raw["validation"]
    if limit:
        train = train.select(range(min(limit, len(train))))

    tokenizer = AutoTokenizer.from_pretrained(rcfg.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(rcfg.model_name)

    train_feats = train.map(lambda x: _prepare_train_features(x, tokenizer, rcfg.max_length, rcfg.doc_stride),
                            batched=True, remove_columns=train.column_names)
    val_feats = val.select(range(min(2000, len(val)))).map(
        lambda x: _prepare_train_features(x, tokenizer, rcfg.max_length, rcfg.doc_stride),
        batched=True, remove_columns=val.column_names)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and rcfg.bf16
    fp16 = (not bf16) and torch.cuda.is_available() and rcfg.fp16
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args = _build_args(rcfg, Path(rcfg.output_dir) / "_ckpt", bf16, fp16)

    trainer = Trainer(model=model, args=args, train_dataset=train_feats, eval_dataset=val_feats,
                      data_collator=default_data_collator, tokenizer=tokenizer)
    ckpt_dir = Path(rcfg.output_dir) / "_ckpt"
    last = get_last_checkpoint(str(ckpt_dir)) if any(ckpt_dir.glob("checkpoint-*")) else None
    logger.info("Training reader %s (bf16=%s) | train feats=%d", rcfg.model_name, bf16, len(train_feats))
    trainer.train(resume_from_checkpoint=last)

    final = Path(rcfg.output_dir) / "latest"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    # EM/F1 eval via the QA pipeline on a validation subset (handles postprocessing)
    metrics = _pipeline_eval(str(final), val, eval_n, rcfg)
    save_model_metadata(final, base_model=rcfg.model_name, task="extractive-qa",
                        config_subset={"epochs": rcfg.num_train_epochs, "lr": rcfg.learning_rate,
                                       "max_length": rcfg.max_length, "doc_stride": rcfg.doc_stride},
                        dataset_info={"dataset": cfg.data.reader_dataset, "train": len(train)},
                        metrics=metrics)
    logger.info("Reader saved -> %s | %s", final, metrics)
    return {"model_dir": str(final), **metrics}


def _pipeline_eval(model_dir, val, n, rcfg) -> Dict:
    from transformers import pipeline
    pipe = pipeline("question-answering", model=model_dir, tokenizer=model_dir, framework="pt", device=-1)
    em = f1 = 0.0
    sub = val.select(range(min(n, len(val))))
    for ex in sub:
        res = pipe(question=ex["question"], context=ex["context"], handle_impossible_answer=True,
                   max_seq_len=rcfg.max_length, doc_stride=rcfg.doc_stride)
        pred = (res.get("answer") or "").strip()
        golds = ex["answers"]["text"]
        gold = golds[0] if golds else ""  # empty = no-answer
        em += float(normalize_answer(pred) == normalize_answer(gold))
        f1 += max([_f1(pred, g) for g in (golds or [""])] or [0.0])
    k = max(1, len(sub))
    return {"exact_match": round(100 * em / k, 2), "f1": round(100 * f1 / k, 2), "eval_n": len(sub)}


__all__ = ["train_reader"]
