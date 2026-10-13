"""Encoder cross-validation with fragment- and play-level evaluation."""

import os
import random

import json
import math
from pathlib import Path
from collections import deque
from typing import List, Optional, Tuple

import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Configuration


# A fixed seed does not ensure bitwise reproducibility on GPU.

REPO_ROOT = Path(__file__).resolve().parents[1]

SEED = 44


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
FOLDS_DIR = str(REPO_ROOT / "data/crossval_12folds_mroberta_tokenized")
OUTPUT_DIR = str(REPO_ROOT / "outputs/crossval_experiments/beto_crossval_12folds")

EARLY_STOPPING_PATIENCE     = 10
NUM_TRAIN_EPOCHS            = 45
PER_DEVICE_TRAIN_BATCH_SIZE = 4
LEARNING_RATE               = 4e-6
METRIC_FOR_BEST             = "eval_accuracy"
GRAD_ACCUM_STEPS            = 4
WARMUP_RATIO                = 0.1
MAX_LENGTH                  = 512
MAX_GRAD_NORM               = 1.0
LOGGING_STEPS               = 50
EVAL_STRATEGY               = "epoch"


# Author labels


id2label = {
    0: "ALARCON",
    1: "CALDERON",
    2: "CASTRO",
    3: "CERVANTES",
    4: "LOPE",
    5: "MONTALBAN",
    6: "MORETO",
    7: "ROJAS",
    8: "TIRSO",
}
label2id = {v: k for k, v in id2label.items()}
NUM_LABELS  = len(id2label)
LABEL_IDS   = list(id2label.keys())
LABEL_NAMES = [id2label[i] for i in LABEL_IDS]


os.makedirs(OUTPUT_DIR, exist_ok=True)

fold_files = sorted(glob.glob(os.path.join(FOLDS_DIR, "*.csv")))
n_folds = len(fold_files)
if n_folds == 0:
    raise ValueError(f"No se encontraron CSVs en {FOLDS_DIR}")
print(f"Encontrados {n_folds} folds.")

with open(os.path.join(OUTPUT_DIR, "folds_selection.txt"), "w", encoding="utf-8") as f:
    for p in fold_files:
        f.write(p + "\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, ids=None, max_length=512):
        self.texts      = texts
        self.labels     = labels.values if hasattr(labels, "values") else labels
        self.ids        = ids if ids is not None else list(range(len(texts)))
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = int(self.labels[idx])
        enc["id"]     = int(self.ids[idx])
        return enc


def collate_with_ids(features):
    """Preserve sample IDs while padding model inputs."""
    ids = [f.pop("id") for f in features]
    batch = base_data_collator(features)
    batch["ids"] = torch.tensor(ids, dtype=torch.long)
    return batch


# Encoder representations before the classification head


def extract_prehead_embeddings(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 16,
    max_length: int = 512,
) -> np.ndarray:
    model.eval()
    base = getattr(model, "base_model", model)
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, truncation=True, padding=True,
                max_length=max_length, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = base(**enc, output_hidden_states=True, return_dict=True)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                rep = out.pooler_output.cpu().numpy()
            else:
                last_hidden = out.hidden_states[-1]
                mask = enc["attention_mask"].unsqueeze(-1).float()
                rep = ((last_hidden * mask).sum(1) / mask.sum(1)).cpu().numpy()
            all_embs.append(rep)
    if not all_embs:
        return np.zeros((0, model.config.hidden_size))
    return np.vstack(all_embs)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Return loss, accuracy and macro-F1."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss, total_examples = 0.0, 0

    for batch in dataloader:
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=labels)
        if outputs.loss is not None:
            total_loss += outputs.loss.item() * labels.size(0)
        total_examples += labels.size(0)
        all_logits.append(outputs.logits.float().detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    if not all_logits:
        return {"loss": None, "accuracy": None, "f1_macro": None}

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    preds      = np.argmax(all_logits, axis=1)

    return {
        "loss":     float(total_loss / total_examples) if total_examples > 0 else None,
        "accuracy": float(accuracy_score(all_labels, preds)),
        "f1_macro": float(f1_score(all_labels, preds, average="macro")),
    }


# Play-level aggregation


def aggregate_by_comedy(
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate fragments by majority vote and summed class probabilities.

    Sample IDs refer to test_df indices; returned probabilities are normalized sums."""

    idx_to_comedy = test_df["comedia"].to_dict()

    records = []
    for sid, true_l, pred_l, prob_v in zip(sample_ids, y_true, y_pred, y_probs):
        comedy_name = idx_to_comedy.get(int(sid), "UNKNOWN")
        records.append({
            "comedia":    comedy_name,
            "true_label": int(true_l),
            "pred_label": int(pred_l),
            "probs":      prob_v,
        })

    chunk_df = pd.DataFrame(records)

    comedy_rows = []
    for comedy_name, grp in chunk_df.groupby("comedia"):
        # Use the modal reference label within each play.
        true_labels_in_group = grp["true_label"].values
        true_label_comedy = int(pd.Series(true_labels_in_group).mode().iloc[0])
        n_chunks = len(grp)

        # Majority vote over fragment predictions.
        pred_counts = pd.Series(grp["pred_label"].values).value_counts()
        hard_pred = int(pred_counts.idxmax())

        # Sum class probabilities across fragments.
        prob_matrix = np.vstack(grp["probs"].values)
        summed_probs = prob_matrix.sum(axis=0)
        soft_pred = int(np.argmax(summed_probs))
        soft_probs_norm = summed_probs / summed_probs.sum()

        comedy_rows.append({
            "comedia":            comedy_name,
            "n_chunks":           n_chunks,
            "true_label":         true_label_comedy,
            "true_label_name":    id2label.get(true_label_comedy, str(true_label_comedy)),
            "hard_pred_label":    hard_pred,
            "hard_pred_name":     id2label.get(hard_pred, str(hard_pred)),
            "soft_pred_label":    soft_pred,
            "soft_pred_name":     id2label.get(soft_pred, str(soft_pred)),
            "hard_correct":       int(hard_pred == true_label_comedy),
            "soft_correct":       int(soft_pred == true_label_comedy),
            "soft_probs":         json.dumps(list(soft_probs_norm), cls=NumpyEncoder),
        })

    comedy_df = pd.DataFrame(comedy_rows)

    comedy_true      = comedy_df["true_label"].values
    comedy_pred_hard = comedy_df["hard_pred_label"].values
    comedy_pred_soft = comedy_df["soft_pred_label"].values
    comedy_probs_soft = np.vstack(
        [json.loads(r) for r in comedy_df["soft_probs"].values]
    ) if len(comedy_df) > 0 else np.empty((0, NUM_LABELS))

    return comedy_df, comedy_true, comedy_pred_hard, comedy_pred_soft, comedy_probs_soft


def comedy_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Return play-level accuracy, macro-F1 and micro-F1."""
    if len(y_true) == 0:
        return {"accuracy": None, "f1_macro": None, "f1_micro": None}
    prefix = f"{label}_" if label else ""
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0)),
        f"{prefix}f1_micro": float(f1_score(y_true, y_pred, average="micro",
                                             zero_division=0)),
    }


def save_comedy_results(
    comedy_df: pd.DataFrame,
    comedy_true: np.ndarray,
    comedy_pred_hard: np.ndarray,
    comedy_pred_soft: np.ndarray,
    comedy_probs_soft: np.ndarray,
    fold_outdir: str,
    fold_idx: int,
) -> dict:
    """Save play-level predictions and return their metrics."""
    comedy_dir = os.path.join(fold_outdir, "comedy_level")
    os.makedirs(comedy_dir, exist_ok=True)


    comedy_df.to_csv(os.path.join(comedy_dir, "predictions_comedy.csv"),
                     index=False, encoding="utf-8")


    for strategy, preds in [("hard", comedy_pred_hard), ("soft", comedy_pred_soft)]:
        if len(comedy_true) == 0:
            continue
        cm = confusion_matrix(comedy_true, preds, labels=LABEL_IDS)
        np.save(os.path.join(comedy_dir, f"confusion_matrix_{strategy}.npy"), cm)
        pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
            os.path.join(comedy_dir, f"confusion_matrix_{strategy}.csv"),
            encoding="utf-8"
        )


    if comedy_probs_soft.size > 0:
        np.save(os.path.join(comedy_dir, "comedy_soft_probs.npy"), comedy_probs_soft)


    metrics_hard = comedy_metrics(comedy_true, comedy_pred_hard, label="hard")
    metrics_soft = comedy_metrics(comedy_true, comedy_pred_soft, label="soft")
    combined = {**metrics_hard, **metrics_soft, "n_comedias": len(comedy_df)}

    with open(os.path.join(comedy_dir, "comedy_metrics.json"), "w", encoding="utf-8") as jf:
        json.dump(combined, jf, indent=2, cls=NumpyEncoder)

    print(f"[Fold {fold_idx}] Métricas por comedia (hard voting): {metrics_hard}")
    print(f"[Fold {fold_idx}] Métricas por comedia (soft voting): {metrics_soft}")
    return combined


@torch.no_grad()
def predict_and_save(
    model, dataloader, device, fold_outdir, fold_idx, test_source_file, test_df
):
    """Save test predictions and return fragment- and play-level results."""
    model.eval()
    all_logits, all_labels, all_ids = [], [], []

    for batch in dataloader:
        ids    = batch.pop("ids").cpu().numpy()
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        all_logits.append(outputs.logits.float().detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_ids.append(ids)

    empty_cm = np.zeros((NUM_LABELS, NUM_LABELS), dtype=int)
    empty_arr = np.array([])
    empty_probs = np.empty((0, NUM_LABELS))

    if not all_logits:
        return (empty_arr, empty_arr, empty_probs, empty_cm,
                empty_arr, empty_arr, empty_arr, empty_probs)

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_ids    = np.concatenate(all_ids)
    probs      = F.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    preds      = np.argmax(all_logits, axis=1)

    # Fragment-level results
    preds_df = pd.DataFrame({
        "sample_idx_in_fold": all_ids,
        "global_sample_id":   [f"{test_source_file}_{int(i)}" for i in all_ids],
        "true_label":         all_labels,
        "true_label_name":    [id2label[int(l)] for l in all_labels],
        "pred_label":         preds,
        "pred_label_name":    [id2label[int(p)] for p in preds],
        "probs":              [json.dumps(list(p), cls=NumpyEncoder) for p in probs],
    })
    preds_csv = os.path.join(fold_outdir, "predictions_test.csv")
    preds_df.to_csv(preds_csv, index=False, encoding="utf-8")

    cm_chunk = confusion_matrix(all_labels, preds, labels=LABEL_IDS)
    np.save(os.path.join(fold_outdir, "confusion_matrix.npy"), cm_chunk)
    pd.DataFrame(cm_chunk, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        os.path.join(fold_outdir, "confusion_matrix.csv"), encoding="utf-8"
    )

    tp = np.diag(cm_chunk)
    fp = cm_chunk.sum(axis=0) - tp
    fn = cm_chunk.sum(axis=1) - tp
    pd.DataFrame({
        "label_id":   LABEL_IDS,
        "label_name": LABEL_NAMES,
        "TP": tp, "FP": fp, "FN": fn,
        "support": cm_chunk.sum(axis=1),
    }).to_csv(os.path.join(fold_outdir, "counts_per_class.csv"), index=False, encoding="utf-8")

    print(f"[Fold {fold_idx}] Predicciones chunk-level guardadas en: {fold_outdir}")

    # Play-level results
    comedy_df_fold, comedy_true, comedy_pred_hard, comedy_pred_soft, comedy_probs_soft = \
        aggregate_by_comedy(all_ids, all_labels, preds, probs, test_df)

    save_comedy_results(
        comedy_df_fold, comedy_true, comedy_pred_hard, comedy_pred_soft,
        comedy_probs_soft, fold_outdir, fold_idx
    )

    return (all_labels, preds, probs, cm_chunk,
            comedy_true, comedy_pred_hard, comedy_pred_soft, comedy_probs_soft)


def save_training_plot(history: dict, fold_idx: int, png_path: str):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.plot(epochs, history["train_loss"], label="train_loss")
    if any(v is not None for v in history["val_loss"]):
        ax1.plot(epochs, [v if v is not None else np.nan for v in history["val_loss"]],
                 label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    for key, marker, style, label in [
        ("val_accuracy",  "o", "--", "val_accuracy"),
        ("val_f1",        "x", "--", "val_f1"),
        ("test_accuracy", "s", ":",  "test_accuracy"),
        ("test_f1",       "d", ":",  "test_f1"),
    ]:
        vals = history.get(key, [])
        if any(v is not None for v in vals):
            ax2.plot(epochs[:len(vals)], [v if v is not None else np.nan for v in vals],
                     linestyle=style, marker=marker, label=label)
    ax2.set_ylabel("Accuracy / F1")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title(f"Training progress (fold {fold_idx})")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(vals)),
        "std":  float(np.std(vals)),
        "min":  float(np.min(vals)),
        "max":  float(np.max(vals)),
    }


# Cross-validation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

results_per_fold = []


global_all_true   = []
global_all_pred   = []
global_all_probs  = []


global_comedy_true_hard  = []
global_comedy_pred_hard  = []
global_comedy_true_soft  = []
global_comedy_pred_soft  = []
global_comedy_probs_soft = []

for fold_idx in range(n_folds):
    # Reset random generators for each fold.
    seed_everything(SEED)
    train_generator = torch.Generator().manual_seed(SEED)
    print("\n" + "=" * 60)
    print(f"ITERACIÓN {fold_idx + 1}/{n_folds}: test = {os.path.basename(fold_files[fold_idx])}")
    print("=" * 60)

    fold_outdir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx:02d}")
    os.makedirs(fold_outdir, exist_ok=True)

    test_file        = fold_files[fold_idx]
    val_idx          = (fold_idx + 1) % n_folds
    val_file         = fold_files[val_idx]
    train_files      = [f for i, f in enumerate(fold_files) if i != fold_idx and i != val_idx]
    test_source_file = os.path.basename(test_file)

    train_df = pd.concat([pd.read_csv(p) for p in train_files], ignore_index=True)
    eval_df  = pd.read_csv(val_file).reset_index(drop=True)
    test_df  = pd.read_csv(test_file).reset_index(drop=True)


    if "comedia" not in test_df.columns:
        raise ValueError(
            f"El CSV {test_file} no contiene la columna 'comedia'. "
            "Es necesaria para la agregación por comedia."
        )

    for df in (train_df, eval_df, test_df):
        df["label"] = df["autor"].map(label2id)

    with open(os.path.join(fold_outdir, "files_used.txt"), "w", encoding="utf-8") as f:
        f.write(f"test_file: {test_file}\nval_file: {val_file}\ntrain_files:\n")
        for p in train_files:
            f.write(f"  {p}\n")


    train_dataset = TextDataset(train_df["texto"].tolist(), train_df["label"], tokenizer,
                                ids=list(train_df.index), max_length=MAX_LENGTH)
    eval_dataset  = TextDataset(eval_df["texto"].tolist(),  eval_df["label"],  tokenizer,
                                ids=list(eval_df.index),  max_length=MAX_LENGTH)
    test_dataset  = TextDataset(test_df["texto"].tolist(),  test_df["label"],  tokenizer,
                                ids=list(test_df.index),  max_length=MAX_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                                  shuffle=True,  collate_fn=base_data_collator,
                                  generator=train_generator)
    eval_dataloader  = DataLoader(eval_dataset,  batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                                  shuffle=False, collate_fn=base_data_collator)
    test_dataloader  = DataLoader(test_dataset,  batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                                  shuffle=False, collate_fn=collate_with_ids)


    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
    ).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ], lr=LEARNING_RATE)

    steps_per_epoch = max(math.ceil(len(train_dataloader) / GRAD_ACCUM_STEPS), 1)
    total_steps     = steps_per_epoch * NUM_TRAIN_EPOCHS
    warmup_steps    = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_metric   = -float("inf")
    best_checkpoint   = None
    epochs_no_improve = 0
    global_step       = 0
    trained_epochs    = 0

    history = {k: [] for k in ("train_loss", "val_loss", "val_accuracy", "val_f1",
                                "test_accuracy", "test_f1")}

    model.zero_grad()
    for epoch in range(1, NUM_TRAIN_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                        desc=f"Fold {fold_idx} Epoch {epoch}", leave=False)

        for step, batch in progress:
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}
            loss   = model(**batch, labels=labels).loss / GRAD_ACCUM_STEPS
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % LOGGING_STEPS == 0:
                progress.set_postfix({"loss": f"{epoch_loss / (step + 1):.4f}",
                                      "step": global_step})

        trained_epochs = epoch
        avg_train_loss = epoch_loss / max(len(train_dataloader), 1)
        history["train_loss"].append(avg_train_loss)

        if EVAL_STRATEGY == "epoch":
            val_m  = evaluate(model, eval_dataloader, device)
            test_m = evaluate(model, test_dataloader, device)
            print(f"  Fold {fold_idx} Ep {epoch} | VAL:  {val_m}")
            print(f"  Fold {fold_idx} Ep {epoch} | TEST: {test_m}")

            history["val_loss"].append(val_m.get("loss"))
            history["val_accuracy"].append(val_m.get("accuracy"))
            history["val_f1"].append(val_m.get("f1_macro"))
            history["test_accuracy"].append(test_m.get("accuracy"))
            history["test_f1"].append(test_m.get("f1_macro"))

            metric_key     = METRIC_FOR_BEST.replace("eval_", "")
            current_metric = val_m.get(metric_key)

            if current_metric is not None:
                if current_metric > best_val_metric:
                    best_val_metric   = current_metric
                    epochs_no_improve = 0
                    best_checkpoint   = os.path.join(fold_outdir, "best_model")
                    os.makedirs(best_checkpoint, exist_ok=True)
                    model.save_pretrained(best_checkpoint)
                    tokenizer.save_pretrained(best_checkpoint)
                    print(f"  ✓ Nuevo mejor modelo (ep {epoch}, "
                          f"{metric_key}={best_val_metric:.6f})")
                else:
                    epochs_no_improve += 1
                    print(f"  No mejora: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

            json_path = os.path.join(fold_outdir, "training_history.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(history, jf, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            save_training_plot(history, fold_idx,
                               os.path.join(fold_outdir, "training_progress.png"))

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping tras {EARLY_STOPPING_PATIENCE} épocas sin mejora.")
                break

    # Final evaluation with the best validation checkpoint
    if best_checkpoint and os.path.isdir(best_checkpoint):
        print(f"[Fold {fold_idx}] Cargando best checkpoint: {best_checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint).to(device)
    else:
        print(f"[Fold {fold_idx}] Usando modelo actual (sin best_checkpoint).")

    best_val_metrics   = evaluate(model, eval_dataloader, device)
    final_test_metrics = evaluate(model, test_dataloader, device)
    print(f"[Fold {fold_idx}] VAL final:  {best_val_metrics}")
    print(f"[Fold {fold_idx}] TEST final: {final_test_metrics}")

    save_training_plot(history, fold_idx, os.path.join(fold_outdir, "training_progress.png"))

    txt_path = os.path.join(fold_outdir, "best_checkpoint_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Best checkpoint metrics and training info\n")
        f.write("=========================================\n\n")
        f.write(f"Best checkpoint: {best_checkpoint or 'N/A'}\n")
        f.write(f"Epochs trained:  {trained_epochs}\n\n")
        f.write("Validation metrics:\n")
        f.write(json.dumps(best_val_metrics, indent=2, cls=NumpyEncoder))
        f.write("\n\nTest metrics:\n")
        f.write(json.dumps(final_test_metrics, indent=2, cls=NumpyEncoder))


    try:
        texts_test = test_df["texto"].fillna("").astype(str).tolist()
        emb = extract_prehead_embeddings(model, tokenizer, texts_test, device,
                                         batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                                         max_length=MAX_LENGTH)
        reps_dir = os.path.join(fold_outdir, "representations")
        os.makedirs(reps_dir, exist_ok=True)
        rep_ids = np.arange(len(emb), dtype=int)
        np.save(os.path.join(reps_dir, "representations.npy"), emb)
        np.save(os.path.join(reps_dir, "rep_ids.npy"), rep_ids)
        pd.DataFrame({
            "sample_idx_in_fold": rep_ids,
            "global_sample_id":   [f"{test_source_file}_{i}" for i in rep_ids],
        }).to_csv(os.path.join(reps_dir, "representations_index.csv"),
                  index=False, encoding="utf-8")
        print(f"[Fold {fold_idx}] Representaciones guardadas en: {reps_dir}")
    except Exception as e:
        print(f"[Fold {fold_idx}] Error guardando representaciones: {e}")


    (y_true, y_pred, y_probs, cm_fold,
     comedy_true, comedy_pred_hard, comedy_pred_soft, comedy_probs_soft) = predict_and_save(
        model, test_dataloader, device, fold_outdir, fold_idx, test_source_file, test_df
    )


    if y_true.size > 0:
        global_all_true.append(y_true)
        global_all_pred.append(y_pred)
        global_all_probs.append(y_probs)


    if comedy_true.size > 0:
        global_comedy_true_hard.append(comedy_true)
        global_comedy_pred_hard.append(comedy_pred_hard)
        global_comedy_true_soft.append(comedy_true)
        global_comedy_pred_soft.append(comedy_pred_soft)
        global_comedy_probs_soft.append(comedy_probs_soft)


    comedy_dir = os.path.join(fold_outdir, "comedy_level")
    comedy_m_path = os.path.join(comedy_dir, "comedy_metrics.json")
    comedy_m = {}
    if os.path.isfile(comedy_m_path):
        with open(comedy_m_path, encoding="utf-8") as jf:
            comedy_m = json.load(jf)

    results_per_fold.append({
        "fold_idx":                    fold_idx,
        "seed":                        SEED,
        "test_file":                   test_source_file,
        "val_file":                    os.path.basename(val_file),
        "trained_epochs":              trained_epochs,

        "best_validation_f1":          best_val_metrics.get("f1_macro"),
        "best_validation_accuracy":    best_val_metrics.get("accuracy"),
        "test_f1_chunk":               final_test_metrics.get("f1_macro"),
        "test_accuracy_chunk":         final_test_metrics.get("accuracy"),

        "test_f1_comedy_hard":         comedy_m.get("hard_f1_macro"),
        "test_accuracy_comedy_hard":   comedy_m.get("hard_accuracy"),

        "test_f1_comedy_soft":         comedy_m.get("soft_f1_macro"),
        "test_accuracy_comedy_soft":   comedy_m.get("soft_accuracy"),
        "n_comedias_test":             comedy_m.get("n_comedias"),
        "best_checkpoint":             best_checkpoint or "N/A",
    })

    del model
    torch.cuda.empty_cache()


summary = {
    "seed": SEED,
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "n_folds":  n_folds,
    "per_fold": results_per_fold,
    # Variation across folds
    "chunk": {
        "val_f1":        mean_std([r["best_validation_f1"]       for r in results_per_fold]),
        "val_accuracy":  mean_std([r["best_validation_accuracy"]  for r in results_per_fold]),
        "test_f1":       mean_std([r["test_f1_chunk"]             for r in results_per_fold]),
        "test_accuracy": mean_std([r["test_accuracy_chunk"]       for r in results_per_fold]),
    },

    "comedy": {
        "hard": {
            "test_f1":       mean_std([r["test_f1_comedy_hard"]       for r in results_per_fold]),
            "test_accuracy": mean_std([r["test_accuracy_comedy_hard"]  for r in results_per_fold]),
        },
        "soft": {
            "test_f1":       mean_std([r["test_f1_comedy_soft"]       for r in results_per_fold]),
            "test_accuracy": mean_std([r["test_accuracy_comedy_soft"]  for r in results_per_fold]),
        },
    },
}

# Pooled fragment predictions
if global_all_true:
    all_true  = np.concatenate(global_all_true)
    all_pred  = np.concatenate(global_all_pred)
    all_probs = np.concatenate(global_all_probs)

    global_cm_chunk = confusion_matrix(all_true, all_pred, labels=LABEL_IDS)
    np.save(os.path.join(OUTPUT_DIR, "confusion_matrix_global.npy"), global_cm_chunk)
    pd.DataFrame(global_cm_chunk, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        os.path.join(OUTPUT_DIR, "confusion_matrix_global.csv"), encoding="utf-8"
    )

    p, r, f, sup = precision_recall_fscore_support(
        all_true, all_pred, labels=LABEL_IDS, zero_division=0
    )
    per_class_csv = os.path.join(OUTPUT_DIR, "per_class_metrics_global.csv")
    pd.DataFrame({
        "label_id": LABEL_IDS, "label_name": LABEL_NAMES,
        "precision": p, "recall": r, "f1": f, "support": sup,
    }).to_csv(per_class_csv, index=False, encoding="utf-8")

    np.save(os.path.join(OUTPUT_DIR, "global_probs.npy"), all_probs)

    summary["chunk"]["pooled"] = {
        "accuracy":              float(accuracy_score(all_true, all_pred)),
        "macro_f1":              float(f1_score(all_true, all_pred, average="macro")),
        "micro_f1":              float(f1_score(all_true, all_pred, average="micro")),
        "per_class_metrics_csv": per_class_csv,
    }
    print("\nMétricas pooled CHUNK:")
    print(json.dumps(summary["chunk"]["pooled"], indent=2, cls=NumpyEncoder))

# Pooled play predictions
if global_comedy_true_hard:
    comedy_dir_global = os.path.join(OUTPUT_DIR, "comedy_level_global")
    os.makedirs(comedy_dir_global, exist_ok=True)


    all_ct_hard = np.concatenate(global_comedy_true_hard)
    all_cp_hard = np.concatenate(global_comedy_pred_hard)

    cm_comedy_hard = confusion_matrix(all_ct_hard, all_cp_hard, labels=LABEL_IDS)
    np.save(os.path.join(comedy_dir_global, "confusion_matrix_comedy_hard.npy"), cm_comedy_hard)
    pd.DataFrame(cm_comedy_hard, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        os.path.join(comedy_dir_global, "confusion_matrix_comedy_hard.csv"), encoding="utf-8"
    )

    p_h, r_h, f_h, sup_h = precision_recall_fscore_support(
        all_ct_hard, all_cp_hard, labels=LABEL_IDS, zero_division=0
    )
    pd.DataFrame({
        "label_id": LABEL_IDS, "label_name": LABEL_NAMES,
        "precision": p_h, "recall": r_h, "f1": f_h, "support": sup_h,
    }).to_csv(os.path.join(comedy_dir_global, "per_class_metrics_comedy_hard.csv"),
              index=False, encoding="utf-8")


    all_ct_soft = np.concatenate(global_comedy_true_soft)
    all_cp_soft = np.concatenate(global_comedy_pred_soft)
    all_cp_probs_soft = np.concatenate(global_comedy_probs_soft)

    cm_comedy_soft = confusion_matrix(all_ct_soft, all_cp_soft, labels=LABEL_IDS)
    np.save(os.path.join(comedy_dir_global, "confusion_matrix_comedy_soft.npy"), cm_comedy_soft)
    pd.DataFrame(cm_comedy_soft, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        os.path.join(comedy_dir_global, "confusion_matrix_comedy_soft.csv"), encoding="utf-8"
    )

    p_s, r_s, f_s, sup_s = precision_recall_fscore_support(
        all_ct_soft, all_cp_soft, labels=LABEL_IDS, zero_division=0
    )
    pd.DataFrame({
        "label_id": LABEL_IDS, "label_name": LABEL_NAMES,
        "precision": p_s, "recall": r_s, "f1": f_s, "support": sup_s,
    }).to_csv(os.path.join(comedy_dir_global, "per_class_metrics_comedy_soft.csv"),
              index=False, encoding="utf-8")

    np.save(os.path.join(comedy_dir_global, "global_comedy_soft_probs.npy"), all_cp_probs_soft)

    summary["comedy"]["pooled"] = {
        "hard": {
            "n_comedias":  int(len(all_ct_hard)),
            "accuracy":    float(accuracy_score(all_ct_hard, all_cp_hard)),
            "macro_f1":    float(f1_score(all_ct_hard, all_cp_hard, average="macro")),
            "micro_f1":    float(f1_score(all_ct_hard, all_cp_hard, average="micro")),
        },
        "soft": {
            "n_comedias":  int(len(all_ct_soft)),
            "accuracy":    float(accuracy_score(all_ct_soft, all_cp_soft)),
            "macro_f1":    float(f1_score(all_ct_soft, all_cp_soft, average="macro")),
            "micro_f1":    float(f1_score(all_ct_soft, all_cp_soft, average="micro")),
        },
    }
    print("\nMétricas pooled COMEDIA – Hard voting:")
    print(json.dumps(summary["comedy"]["pooled"]["hard"], indent=2, cls=NumpyEncoder))
    print("\nMétricas pooled COMEDIA – Soft voting:")
    print(json.dumps(summary["comedy"]["pooled"]["soft"], indent=2, cls=NumpyEncoder))
else:
    print("No se acumularon predicciones de comedia.")


summary_json = os.path.join(OUTPUT_DIR, "crossval_summary.json")
with open(summary_json, "w", encoding="utf-8") as jf:
    json.dump(summary, jf, indent=2, ensure_ascii=False, cls=NumpyEncoder)

pd.DataFrame(results_per_fold).to_csv(
    os.path.join(OUTPUT_DIR, "crossval_summary.csv"), index=False
)

print("\nCross-validation terminado.")
print("Resumen guardado en:", summary_json)
