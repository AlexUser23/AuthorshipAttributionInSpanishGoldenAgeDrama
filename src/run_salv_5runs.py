import os
import json
import math
from pathlib import Path
from collections import deque

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Use a non-interactive plotting backend.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed
)

from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


REPO_ROOT = Path(__file__).resolve().parents[1]


# Configuration

MODEL_NAME = "BSC-LT/MrBERT-es"
TRAIN_CSV = str(REPO_ROOT / "data/salv_splits_seed44/train.csv")
EVAL_CSV = str(REPO_ROOT / "data/salv_splits_seed44/val.csv")
TEST_CSV = str(REPO_ROOT / "data/salv_splits_seed44/test.csv")
DUDA_CSV = str(REPO_ROOT / "data/salv_splits_seed44/duda.csv")
OUTPUT_DIR = str(REPO_ROOT / "outputs/salv_experiments/mrbert_es_salv_5runs")

SEEDS = (1, 2, 3, 4, 5)


EARLY_STOPPING_PATIENCE = 10
NUM_TRAIN_EPOCHS = 45
PER_DEVICE_TRAIN_BATCH_SIZE = 4
LEARNING_RATE = 1e-5
SAVE_TOTAL_LIMIT = 2
METRIC_FOR_BEST = "eval_accuracy"
GRAD_ACCUM_STEPS = 4
WARMUP_RATIO = 0.1
MAX_LENGTH = 512
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 50
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"


# PCA dimensions before UMAP.
PCA_DIMS_FOR_UMAP = 300


id2author = id2label = {
    0: "Alarcón",
    1: "Belmonte",
    2: "Claramonte",
    3: "Lope",
    4: "Tirso",
}

author2id = label2id = {
    "Alarcón":    0,
    "Belmonte":   1,
    "Claramonte": 2,
    "Lope":       3,
    "Tirso":      4,
}


AUTHOR_COLORS = {
    "Alarcón":    "#E63946",
    "Belmonte":   "#2A9D8F",
    "Claramonte": "#F4A261",
    "Lope":       "#457B9D",
    "Tirso":      "#8338EC",
    "DUDA":       "#AAAAAA",
}


MARKER_STYLES = ["o", "s", "^", "D", "v", "P", "*", "X", "h", ">", "<", "p", "H", "8", "+"]

def run_experiment(SEED, OUTPUT_DIR):

    set_seed(SEED)


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


    train_df = pd.read_csv(TRAIN_CSV)
    eval_df  = pd.read_csv(EVAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)


    duda_df = pd.read_csv(DUDA_CSV)
    print(f"[INFO] Cargados {len(duda_df)} chunk(s) de la comedia dudosa desde: {DUDA_CSV}")
    print(f"[INFO] Comedias DUDA: {duda_df['comedia'].unique().tolist()}")


    train_df["label"]  = train_df["autor"].map(author2id)
    eval_df["label"]   = eval_df["autor"].map(author2id)
    test_df["label"]   = test_df["autor"].map(author2id)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(author2id),
        id2label=id2author,
        label2id=author2id
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=512):
            self.texts      = texts
            self.labels     = labels.values if hasattr(labels, "values") else labels
            self.tokenizer  = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text  = str(self.texts[idx])
            label = int(self.labels[idx])
            enc   = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
            enc["labels"] = label
            return enc


    class TextDatasetNoLabel(Dataset):
        """Unlabelled fragments of the disputed play."""
        def __init__(self, texts, tokenizer, max_length=512):
            self.texts      = texts
            self.tokenizer  = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            enc  = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
            return enc


    train_dataset = TextDataset(train_df["texto"].tolist(), train_df["label"], tokenizer, max_length=MAX_LENGTH)
    eval_dataset  = TextDataset(eval_df["texto"].tolist(),  eval_df["label"],  tokenizer, max_length=MAX_LENGTH)
    test_dataset  = TextDataset(test_df["texto"].tolist(),  test_df["label"],  tokenizer, max_length=MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_dataloader = DataLoader(train_dataset, batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, shuffle=True,  collate_fn=data_collator)
    eval_dataloader  = DataLoader(eval_dataset,  batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, shuffle=False, collate_fn=data_collator)
    test_dataloader  = DataLoader(test_dataset,  batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, shuffle=False, collate_fn=data_collator)


    def compute_metrics_from_logits(logits, labels):
        preds = np.argmax(logits, axis=1)
        acc   = accuracy_score(labels, preds)
        f1    = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}


    def compute_comedy_majority_vote_metrics(df_known, chunk_preds, split_name=""):
        df = df_known.copy()
        df["pred"] = chunk_preds

        comedy_true, comedy_pred = [], []
        for comedia, grp in df.groupby("comedia"):
            true_autor = grp["autor"].iloc[0]
            voted      = grp["pred"].value_counts().idxmax()
            comedy_true.append(true_autor)
            comedy_pred.append(voted)

        acc = accuracy_score(comedy_true, comedy_pred)
        f1  = f1_score(comedy_true, comedy_pred, average="macro",
                       labels=list(author2id.keys()), zero_division=0)

        print(f"\n[COMEDY-VOTE] {split_name} — a nivel de comedia ({len(comedy_true)} comedias):")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-macro : {f1:.4f}")
        for ct, cp in zip(comedy_true, comedy_pred):
            status = "✓" if ct == cp else "✗"
            print(f"    {status}  real={ct:12s}  pred={cp}")

        return {
            "comedy_accuracy": acc,
            "comedy_f1_macro": f1,
            "comedy_true":     comedy_true,
            "comedy_pred":     comedy_pred,
        }


    def save_confusion_matrices(y_true_chunk, y_pred_chunk,
                                 y_true_comedy, y_pred_comedy,
                                 label_names, split_name, output_dir):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        cm_chunk = confusion_matrix(y_true_chunk, y_pred_chunk,
                                    labels=list(range(len(label_names))))
        disp_chunk = ConfusionMatrixDisplay(confusion_matrix=cm_chunk,
                                            display_labels=label_names)
        disp_chunk.plot(ax=axes[0], colorbar=False, xticks_rotation=45)
        axes[0].set_title(f"{split_name} — Chunks (nivel de fragmento)", fontsize=11)

        cm_comedy = confusion_matrix(y_true_comedy, y_pred_comedy,
                                      labels=label_names)
        disp_comedy = ConfusionMatrixDisplay(confusion_matrix=cm_comedy,
                                              display_labels=label_names)
        disp_comedy.plot(ax=axes[1], colorbar=False, xticks_rotation=45)
        axes[1].set_title(f"{split_name} — Comedias (votación mayoritaria)", fontsize=11)

        plt.suptitle(f"Matrices de confusión — {split_name}", fontsize=13, y=1.01)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"confusion_matrices_{split_name.lower()}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[CM] Matrices de confusión ({split_name}) guardadas en: {out_path}")
        return out_path


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRAD_ACCUM_STEPS)
    total_training_steps       = num_update_steps_per_epoch * NUM_TRAIN_EPOCHS
    warmup_steps               = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)


    @torch.no_grad()
    def evaluate(model, dataloader, device):
        model.eval()
        all_logits, all_labels = [], []
        total_loss, total_examples = 0.0, 0

        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device)
            outputs = model(**batch, labels=labels)
            loss    = outputs.loss if hasattr(outputs, "loss") else None
            logits  = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if loss is not None:
                total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            all_logits.append(logits.detach().float().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        avg_loss = (total_loss / total_examples) if total_examples > 0 and total_loss != 0 else None
        if len(all_logits) == 0:
            return {"loss": avg_loss, "accuracy": None, "f1_macro": None,
                    "all_logits": None, "all_labels": None}
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        metrics = compute_metrics_from_logits(all_logits, all_labels)
        metrics["loss"]       = avg_loss
        metrics["all_logits"] = all_logits
        metrics["all_labels"] = all_labels
        return metrics


    # First-token representations from the final hidden layer

    @torch.no_grad()
    def extract_embeddings(model, dataloader, device):
        model.eval()
        all_embeddings = []
        for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            batch.pop("labels", None)
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device)
            outputs  = model(**batch, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            cls_emb  = last_hidden[:, 0, :].detach().float().cpu().numpy()
            all_embeddings.append(cls_emb)
        return np.concatenate(all_embeddings, axis=0)


    @torch.no_grad()
    def predict_duda(model, texts, tokenizer, device, batch_size=4):
        """Return disputed-play predictions, representations and logits."""
        model.eval()
        duda_dataset      = TextDatasetNoLabel(texts, tokenizer, max_length=MAX_LENGTH)
        collator_no_label = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        duda_dataloader   = DataLoader(duda_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator_no_label)

        all_logits, all_embeddings = [], []
        for batch in tqdm(duda_dataloader, desc="Predicting DUDA chunks", leave=False):
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device)
            outputs     = model(**batch, output_hidden_states=True)
            logits      = outputs.logits.detach().float().cpu().numpy()
            last_hidden = outputs.hidden_states[-1]
            cls_emb     = last_hidden[:, 0, :].detach().float().cpu().numpy()
            all_logits.append(logits)
            all_embeddings.append(cls_emb)

        all_logits     = np.concatenate(all_logits, axis=0)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        pred_ids       = np.argmax(all_logits, axis=1)
        pred_authors   = [id2author[i] for i in pred_ids]
        return pred_authors, all_embeddings, all_logits


    history = {
        "train_loss":    [],
        "val_loss":      [],
        "val_accuracy":  [],
        "val_f1":        [],
        "test_accuracy": [],
        "test_f1":       []
    }


    # Training

    best_val_metric   = -float("inf")
    best_checkpoint   = None
    epochs_no_improve = 0
    saved_checkpoints = deque()

    global_step = 0
    model.zero_grad()

    print("Comienzo de entrenamiento en device:", device)
    for epoch in range(1, NUM_TRAIN_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        progress   = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}", leave=False)
        optimizer.zero_grad()
        for step, batch in progress:
            labels = batch.pop("labels").to(device)
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device)
            outputs = model(**batch, labels=labels)
            loss    = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % LOGGING_STEPS == 0:
                progress.set_postfix({"loss": f"{(epoch_loss/(step+1)):.4f}", "step": global_step})

        avg_train_loss = epoch_loss / max(len(train_dataloader), 1)
        history["train_loss"].append(avg_train_loss)

        if EVAL_STRATEGY == "epoch":
            val_metrics  = evaluate(model, eval_dataloader, device)
            print(f"Epoch {epoch} VAL metrics:", {k: v for k, v in val_metrics.items() if k not in ("all_logits", "all_labels")})
            history["val_loss"].append(val_metrics.get("loss", None))
            history["val_accuracy"].append(val_metrics.get("accuracy", None))
            history["val_f1"].append(val_metrics.get("f1_macro", None))

            test_metrics = evaluate(model, test_dataloader, device)
            print(f"Epoch {epoch} TEST metrics:", {k: v for k, v in test_metrics.items() if k not in ("all_logits", "all_labels")})
            history["test_accuracy"].append(test_metrics.get("accuracy", None))
            history["test_f1"].append(test_metrics.get("f1_macro", None))

            if SAVE_STRATEGY == "epoch":
                ckpt_name = f"checkpoint-epo{epoch}"
                ckpt_path = os.path.join(OUTPUT_DIR, ckpt_name)
                os.makedirs(ckpt_path, exist_ok=True)
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                saved_checkpoints.append(ckpt_path)
                while len(saved_checkpoints) > SAVE_TOTAL_LIMIT:
                    old = saved_checkpoints.popleft()
                    try:
                        import shutil
                        shutil.rmtree(old)
                    except Exception as e:
                        print(f"Warning: no se pudo eliminar checkpoint antiguo {old}: {e}")

            metric_key     = METRIC_FOR_BEST.replace("eval_", "")
            current_metric = val_metrics.get("f1_macro" if metric_key == "f1_macro" else metric_key, None)

            if current_metric is not None:
                if current_metric > best_val_metric:
                    best_val_metric   = current_metric
                    epochs_no_improve = 0
                    best_checkpoint   = os.path.join(OUTPUT_DIR, "best_model")
                    os.makedirs(best_checkpoint, exist_ok=True)
                    model.save_pretrained(best_checkpoint)
                    tokenizer.save_pretrained(best_checkpoint)
                    print(f"Nuevo mejor modelo (epoch {epoch}) guardado en: {best_checkpoint} (mejor {metric_key}={best_val_metric:.6f})")
                else:
                    epochs_no_improve += 1
                    print(f"No mejora en '{metric_key}'. epochs_no_improve = {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")


            png_path  = os.path.join(OUTPUT_DIR, "training_progress.png")
            json_path = os.path.join(OUTPUT_DIR, "training_history.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(history, jf, indent=2, ensure_ascii=False)

            epochs_x = list(range(1, len(history["train_loss"]) + 1))
            plt.figure(figsize=(9, 6))
            ax1 = plt.gca()
            ax1.plot(epochs_x, history["train_loss"], label="train_loss")
            if any(v is not None for v in history["val_loss"]):
                ax1.plot(epochs_x, [v if v is not None else np.nan for v in history["val_loss"]], label="val_loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax2 = ax1.twinx()
            for key, ls, mk in [("val_accuracy","--","o"), ("val_f1","--","x"), ("test_accuracy",":",  "s"), ("test_f1",":", "d")]:
                if any(v is not None for v in history[key]):
                    ax2.plot(epochs_x, [v if v is not None else np.nan for v in history[key]], linestyle=ls, marker=mk, label=key)
            ax2.set_ylabel("Accuracy / F1")
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
            plt.title("Training progress")
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()
            print(f"Gráfica y JSON de historial guardados en: {png_path}, {json_path}")

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping: no hubo mejora en {EARLY_STOPPING_PATIENCE} épocas. Terminando entrenamiento.")
                break


    if best_checkpoint is not None and os.path.isdir(best_checkpoint):
        print("Cargando mejor checkpoint para evaluación final:", best_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint).to(device)
    else:
        print("No se encontró best_checkpoint; se utilizará el modelo actual.")


    # Final evaluation

    best_val_metrics = evaluate(model, eval_dataloader, device)
    print("Evaluación en VALIDACIÓN (mejor modelo):", {k: v for k, v in best_val_metrics.items() if k not in ("all_logits", "all_labels")})

    test_metrics = evaluate(model, test_dataloader, device)
    print("Evaluación en TEST (final):", {k: v for k, v in test_metrics.items() if k not in ("all_logits", "all_labels")})

    trained_epochs = epoch if 'epoch' in locals() else NUM_TRAIN_EPOCHS

    if len(history["test_accuracy"]) == 0 or len(history["test_accuracy"]) < len(history["train_loss"]):
        history["test_accuracy"].append(test_metrics.get("accuracy", None))
        history["test_f1"].append(test_metrics.get("f1_macro", None))


    label_names = [id2author[i] for i in range(len(id2author))]


    val_chunk_preds      = np.argmax(best_val_metrics["all_logits"], axis=1)
    val_chunk_labels     = best_val_metrics["all_labels"]
    val_chunk_pred_names = [id2author[i] for i in val_chunk_preds]

    eval_df_reset = eval_df.reset_index(drop=True)
    val_comedy_metrics = compute_comedy_majority_vote_metrics(
        eval_df_reset, val_chunk_pred_names, split_name="VAL"
    )

    save_confusion_matrices(
        y_true_chunk  = val_chunk_labels,
        y_pred_chunk  = val_chunk_preds,
        y_true_comedy = val_comedy_metrics["comedy_true"],
        y_pred_comedy = val_comedy_metrics["comedy_pred"],
        label_names   = label_names,
        split_name    = "VAL",
        output_dir    = OUTPUT_DIR
    )


    test_chunk_preds      = np.argmax(test_metrics["all_logits"], axis=1)
    test_chunk_labels     = test_metrics["all_labels"]
    test_chunk_pred_names = [id2author[i] for i in test_chunk_preds]

    test_comedy_metrics = compute_comedy_majority_vote_metrics(
        test_df.reset_index(drop=True), test_chunk_pred_names, split_name="TEST"
    )

    save_confusion_matrices(
        y_true_chunk  = test_chunk_labels,
        y_pred_chunk  = test_chunk_preds,
        y_true_comedy = test_comedy_metrics["comedy_true"],
        y_pred_comedy = test_comedy_metrics["comedy_pred"],
        label_names   = label_names,
        split_name    = "TEST",
        output_dir    = OUTPUT_DIR
    )


    png_path  = os.path.join(OUTPUT_DIR, "training_progress.png")
    json_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(history, jf, indent=2, ensure_ascii=False)

    epochs_x = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(9, 6))
    ax1 = plt.gca()
    ax1.plot(epochs_x, history["train_loss"], label="train_loss")
    if any(v is not None for v in history["val_loss"]):
        ax1.plot(epochs_x, [v if v is not None else np.nan for v in history["val_loss"]], label="val_loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    for key, ls, mk in [("val_accuracy","--","o"),("val_f1","--","x"),("test_accuracy",":","s"),("test_f1",":","d")]:
        if any(v is not None for v in history[key]):
            ax2.plot(epochs_x, [v if v is not None else np.nan for v in history[key]], linestyle=ls, marker=mk, label=key)
    ax2.set_ylabel("Accuracy / F1")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    plt.title("Training progress (final)")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


    # Representation visualizations


    print("\n[VIZ] Extrayendo embeddings del conjunto de test (chunks conocidos)...")

    test_dataset_viz = TextDataset(
        test_df["texto"].tolist(), test_df["label"], tokenizer, max_length=MAX_LENGTH
    )
    test_dataloader_viz = DataLoader(
        test_dataset_viz, batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, shuffle=False, collate_fn=data_collator
    )
    emb_known = extract_embeddings(model, test_dataloader_viz, device)


    authors_known  = test_df["autor"].tolist()
    comedias_known = test_df["comedia"].tolist()


    print(f"[VIZ] Prediciendo autor para {len(duda_df)} chunk(s) de la comedia dudosa...")
    duda_predicted_authors, emb_duda, duda_logits = predict_duda(
        model, duda_df["texto"].tolist(), tokenizer, device,
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE
    )
    duda_comedias = duda_df["comedia"].tolist()


    duda_preds_df = duda_df.copy()
    duda_preds_df["predicted_autor"] = duda_predicted_authors
    duda_preds_path = os.path.join(OUTPUT_DIR, "duda_predictions.csv")
    duda_preds_df.to_csv(duda_preds_path, index=False)
    print(f"[VIZ] Predicciones para chunks DUDA guardadas en: {duda_preds_path}")


    duda_vote = pd.Series(duda_predicted_authors).value_counts()
    duda_voted_author = duda_vote.idxmax()
    print(f"\n[DUDA] Votación mayoritaria sobre la comedia dudosa:")
    print(f"  Autor predicho (mayoría): {duda_voted_author}")
    print(f"  Distribución de votos por chunk:")
    for autor, count in duda_vote.items():
        pct = count / len(duda_predicted_authors) * 100
        print(f"    {autor:12s}: {count:3d} chunks  ({pct:.1f}%)")


    all_embeddings = np.concatenate([emb_known, emb_duda], axis=0)
    all_authors    = authors_known + ["DUDA"] * len(duda_df)
    all_comedias   = comedias_known + duda_comedias

    print(f"[VIZ] Total embeddings: {len(all_embeddings)} ({len(emb_known)} conocidos + {len(emb_duda)} DUDA)")


    unique_comedias = sorted(set(all_comedias))
    comedia2marker  = {c: MARKER_STYLES[i % len(MARKER_STYLES)] for i, c in enumerate(unique_comedias)}


    colors = []
    for a in all_authors:
        matched = next((k for k in AUTHOR_COLORS if k.upper() == a.upper()), None)
        colors.append(AUTHOR_COLORS[matched] if matched else AUTHOR_COLORS.get("DUDA", "#000000"))


    def make_embedding_plot(coords_2d, title, filename, all_authors, all_comedias, colors, comedia2marker):
        fig, ax = plt.subplots(figsize=(14, 10))

        for i, (x, y) in enumerate(coords_2d):
            ax.scatter(
                x, y,
                c=colors[i],
                marker=comedia2marker[all_comedias[i]],
                s=60,
                alpha=0.75,
                edgecolors="none"
            )


        seen_authors = set()
        author_handles = []
        for a in all_authors:
            matched = next((k for k in AUTHOR_COLORS if k.upper() == a.upper()), None)
            key     = matched if matched else a
            if key not in seen_authors:
                seen_authors.add(key)
                color = AUTHOR_COLORS.get(matched, AUTHOR_COLORS.get("DUDA", "#000000"))
                author_handles.append(mpatches.Patch(color=color, label=key))
        author_handles.sort(key=lambda h: h.get_label())


        comedia_handles = [
            Line2D([0], [0], marker=comedia2marker[c], color="gray",
                   linestyle="None", markersize=7, label=c)
            for c in unique_comedias
        ]

        legend1 = ax.legend(handles=author_handles, title="Autor", loc="upper left",
                            bbox_to_anchor=(1.01, 1), fontsize=7, title_fontsize=8)
        legend2 = ax.legend(handles=comedia_handles, title="Comedia", loc="upper left",
                            bbox_to_anchor=(1.01, 0.5), fontsize=6, title_fontsize=7)
        ax.add_artist(legend1)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[VIZ] Guardado: {filename}")


    print("[VIZ] Calculando PCA...")
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(all_embeddings)

    make_embedding_plot(
        coords_pca,
        title="Chunks de test – PCA (colores=autor, formas=comedia)",
        filename=os.path.join(OUTPUT_DIR, "embeddings_pca.png"),
        all_authors=all_authors,
        all_comedias=all_comedias,
        colors=colors,
        comedia2marker=comedia2marker
    )


    try:
        import umap
        print(f"[VIZ] Calculando UMAP sobre reducción PCA ({PCA_DIMS_FOR_UMAP} componentes)...")

        n_pca_umap = min(PCA_DIMS_FOR_UMAP, all_embeddings.shape[1], all_embeddings.shape[0] - 1)
        pca_pre    = PCA(n_components=n_pca_umap, random_state=42)
        emb_pca_pre = pca_pre.fit_transform(all_embeddings)

        reducer     = umap.UMAP(n_neighbors=50, n_components=2, random_state=42, metric="cosine")
        coords_umap = reducer.fit_transform(emb_pca_pre)

        make_embedding_plot(
            coords_umap,
            title=f"Chunks de test – PCA({n_pca_umap}) + UMAP(nn=50) (colores=autor, formas=comedia)",
            filename=os.path.join(OUTPUT_DIR, "embeddings_pca_umap.png"),
            all_authors=all_authors,
            all_comedias=all_comedias,
            colors=colors,
            comedia2marker=comedia2marker
        )
    except ImportError:
        print("[VIZ] WARNING: 'umap-learn' no está instalado. Saltando el plot PCA+UMAP.")
        print("       Instálalo con: pip install umap-learn")


    print("[VIZ] Calculando dendrograma de comedias...")


    comedias_known_arr = np.array(comedias_known)
    emb_known_arr      = emb_known

    comedia_centroids   = {}
    comedia_authors_map = {}
    for comedia in sorted(set(comedias_known)):
        mask = comedias_known_arr == comedia
        comedia_centroids[comedia]   = emb_known_arr[mask].mean(axis=0)
        autor_counts = pd.Series(np.array(authors_known)[mask]).value_counts()
        comedia_authors_map[comedia] = autor_counts.idxmax()


    for comedia in sorted(set(duda_comedias)):
        duda_mask = np.array(duda_comedias) == comedia
        comedia_centroids[comedia]   = emb_duda[duda_mask].mean(axis=0)
        comedia_authors_map[comedia] = "DUDA"

    comedia_names   = sorted(comedia_centroids.keys())
    centroid_matrix = np.array([comedia_centroids[c] for c in comedia_names])

    linkage_matrix = linkage(centroid_matrix, method="ward", metric="euclidean")

    fig, ax = plt.subplots(figsize=(max(12, len(comedia_names) * 0.55), 8))
    ddata = dendrogram(
        linkage_matrix,
        labels=comedia_names,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=0
    )
    plt.title("Dendrograma de comedias (centroides de chunks de test + DUDA)", fontsize=13)
    ax.set_xlabel("Comedia")
    ax.set_ylabel("Distancia (Ward)")

    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        comedia_name = lbl.get_text()
        raw_autor    = comedia_authors_map.get(comedia_name, "")
        matched      = next((k for k in AUTHOR_COLORS if k.upper() == raw_autor.upper()), None)
        lbl.set_color(AUTHOR_COLORS[matched] if matched else "#000000")

    author_handles = []
    seen = set()
    for a in sorted(set(comedia_authors_map.values())):
        matched = next((k for k in AUTHOR_COLORS if k.upper() == a.upper()), None)
        key = matched if matched else a
        if key not in seen:
            color = AUTHOR_COLORS.get(matched, AUTHOR_COLORS.get("DUDA", "#AAAAAA"))
            author_handles.append(mpatches.Patch(color=color, label=key))
            seen.add(key)
    ax.legend(handles=author_handles, title="Autor dominante", loc="upper right", fontsize=8)

    plt.tight_layout()
    dendro_path = os.path.join(OUTPUT_DIR, "dendrogram_comedias.png")
    plt.savefig(dendro_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Dendrograma guardado en: {dendro_path}")


    output_info = {
        "best_checkpoint":           best_checkpoint if best_checkpoint is not None else "No checkpoint registrado",
        "trained_epochs":            trained_epochs,
        "best_validation_metrics":   {k: v for k, v in best_val_metrics.items() if k not in ("all_logits", "all_labels")},
        "test_metrics":              {k: v for k, v in test_metrics.items() if k not in ("all_logits", "all_labels")},
        "val_comedy_vote_accuracy":  val_comedy_metrics["comedy_accuracy"],
        "val_comedy_vote_f1_macro":  val_comedy_metrics["comedy_f1_macro"],
        "test_comedy_vote_accuracy": test_comedy_metrics["comedy_accuracy"],
        "test_comedy_vote_f1_macro": test_comedy_metrics["comedy_f1_macro"],
        "duda_voted_author":         duda_voted_author,
        "duda_vote_distribution":    duda_vote.to_dict(),
        "history_file":              json_path,
        "plot_file":                 png_path,
    }

    txt_path = os.path.join(OUTPUT_DIR, "best_checkpoint_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Best checkpoint metrics and training info\n")
        f.write("=========================================\n\n")
        f.write(f"Best checkpoint path: {output_info['best_checkpoint']}\n")
        f.write(f"Trained epochs (approx): {output_info['trained_epochs']}\n\n")

        f.write("--- Chunk-level metrics (best model) ---\n")
        f.write("Validation metrics (JSON):\n")
        f.write(json.dumps(output_info["best_validation_metrics"], indent=2, ensure_ascii=False))
        f.write("\n\nTest metrics (final evaluation, JSON):\n")
        f.write(json.dumps(output_info["test_metrics"], indent=2, ensure_ascii=False))

        f.write("\n\n--- Comedy-level metrics (majority vote) ---\n")
        f.write(f"VAL  comedy accuracy : {output_info['val_comedy_vote_accuracy']:.4f}\n")
        f.write(f"VAL  comedy F1-macro : {output_info['val_comedy_vote_f1_macro']:.4f}\n")
        f.write(f"TEST comedy accuracy : {output_info['test_comedy_vote_accuracy']:.4f}\n")
        f.write(f"TEST comedy F1-macro : {output_info['test_comedy_vote_f1_macro']:.4f}\n")

        f.write("\n--- Comedia dudosa (DUDA) ---\n")
        f.write(f"Chunks totales DUDA  : {len(duda_df)}\n")
        f.write(f"Autor predicho (voto mayoritario): {duda_voted_author}\n")
        f.write("Distribución de votos por chunk:\n")
        for autor, count in duda_vote.items():
            pct = count / len(duda_predicted_authors) * 100
            f.write(f"  {autor:12s}: {count:3d} chunks  ({pct:.1f}%)\n")

        f.write(f"\nTraining history JSON: {output_info['history_file']}\n")
        f.write(f"Training progress plot: {output_info['plot_file']}\n")

    print(f"\nMétricas y info guardadas en: {txt_path}")
    print("Evaluación en VALIDACIÓN (mejor modelo):", output_info["best_validation_metrics"])
    print("Evaluación en TEST (final):", output_info["test_metrics"])
    print(f"Comedy-vote VAL  → accuracy={output_info['val_comedy_vote_accuracy']:.4f}  F1={output_info['val_comedy_vote_f1_macro']:.4f}")
    print(f"Comedy-vote TEST → accuracy={output_info['test_comedy_vote_accuracy']:.4f}  F1={output_info['test_comedy_vote_f1_macro']:.4f}")
    print(f"DUDA → autor predicho por votación mayoritaria: {duda_voted_author}")
    print("Código terminado")


    metrics = {}
    for split, values in (("val", best_val_metrics), ("test", test_metrics)):
        for metric in ("loss", "accuracy", "f1_macro"):
            metrics[f"{split}_{metric}"] = float(values[metric])
    for key in ("val_comedy_vote_accuracy", "val_comedy_vote_f1_macro",
                "test_comedy_vote_accuracy", "test_comedy_vote_f1_macro"):
        metrics[key] = float(output_info[key])
    metrics["trained_epochs"] = int(trained_epochs)
    # Include zero-vote authors in the run averages.
    duda_results = []
    for comedia, group in duda_preds_df.groupby("comedia", sort=True):
        counts = group["predicted_autor"].value_counts()
        winner = str(counts.idxmax())
        for author in author2id:
            votes = int(counts.get(author, 0))
            duda_results.append({"comedia": str(comedia), "autor": author,
                                 "votos": votes, "porcentaje": 100.0 * votes / len(group),
                                 "ganador": author == winner})
    with open(Path(OUTPUT_DIR) / "run_metrics.json", "w", encoding="utf-8") as stream:
        json.dump({"seed": SEED, "metrics": metrics, "duda": duda_results},
                  stream, ensure_ascii=False, indent=2, allow_nan=False)


def aggregate_runs(results, output_dir):
    """Summarize the five seeds using means and sample standard deviations (ddof=1)."""
    if len(results) != len(SEEDS) or {r["seed"] for r in results} != set(SEEDS):
        raise ValueError("El resumen requiere las cinco semillas completas y distintas.")
    output_dir = Path(output_dir)
    runs = pd.DataFrame([{"seed": r["seed"], **r["metrics"]} for r in results])
    runs.to_csv(output_dir / "metricas_por_run.csv", index=False)
    summary = runs.drop(columns="seed").agg(["mean", "std"]).T
    summary.columns = ["media", "desviacion_tipica"]
    summary.index.name = "metrica"
    summary["n_runs"] = len(results)
    summary.to_csv(output_dir / "resumen_metricas.csv")
    duda = pd.DataFrame([{"seed": r["seed"], **row}
                         for r in results for row in r["duda"]])
    duda.to_csv(output_dir / "duda_por_run.csv", index=False)
    duda_summary = duda.groupby(["comedia", "autor"], sort=True).agg(
        votos_media=("votos", "mean"), votos_std=("votos", "std"),
        porcentaje_media=("porcentaje", "mean"), porcentaje_std=("porcentaje", "std"),
        runs_ganador=("ganador", "sum"), n_runs=("seed", "nunique"))
    duda_summary.to_csv(output_dir / "resumen_duda.csv")
    report = ("Cinco runs independientes; mismas particiones; semillas 1, 2, 3, 4, 5.\n"
              "Desviacion tipica muestral (ddof=1). Accuracy y F1 en escala 0-1.\n"
              "DUDA: porcentaje de votos de fragmentos, no probabilidad de autoria.\n\n"
              + summary.to_string() + "\n\nDUDA\n" + duda_summary.to_string() + "\n")
    (output_dir / "resumen_5_runs.txt").write_text(report, encoding="utf-8")
    print(report)


def main():
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description="Cinco runs de MrBERT con semillas 1 a 5")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Uso interno: ejecutar una semilla")
    args = parser.parse_args()
    root = Path(OUTPUT_DIR)
    if args.seed is not None:
        run_experiment(args.seed, str(root / f"seed_{args.seed}"))
        return


    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"La carpeta de resultados ya contiene archivos: {root}")
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in SEEDS:
        print(f"\n=== Run semilla {seed} ({SEEDS.index(seed) + 1}/{len(SEEDS)}) ===", flush=True)
        # Run each seed in a separate process to release GPU resources.
        with (root / f"seed_{seed}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                            "--seed", str(seed)], stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        with (root / f"seed_{seed}" / "run_metrics.json").open(encoding="utf-8") as stream:
            results.append(json.load(stream))
        print(f"Run {seed} terminado. Resultados: {root / f'seed_{seed}'}", flush=True)
    aggregate_runs(results, root)


if __name__ == "__main__":
    main()
