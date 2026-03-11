"""
Webshell Detector v2 — Optimized Retraining Script
Optimized for "basic laptop" execution:
- Consolidates original data + dataC
- Early Stopping (prevents overtraining and saves time)
- Sequence Length Capping (saves RAM/CPU)
- Relative Path Resolution
- Automatic Device Detection (CPU/CUDA)
- Model Versioning (saves *_v2.pth)
"""

import logging
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.data_loader import load_multiple_datasets
from src.models.model import TextCNN, TextRNN, TransformerClassifier, WebshellDetector

# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────
# Root of the project: src/models/train.py -> src/models -> src -> project_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "models"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer_v2.pkl"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Combined dataset paths
DATASETS = [
    (DATA_DIR / "raw" / "malicious", DATA_DIR / "raw" / "normal"),
    (DATA_DIR / "raw" / "dataC" / "malicious", DATA_DIR / "raw" / "dataC" / "normal"),
]

# Hyperparameters loaded from configs/train_config.yaml
MAX_LEN = config["dataset"]["max_length"]
BATCH_SIZE = config["training"]["batch_size"]
EMBEDDING_DIM = config["word2vec"]["vector_size"]
EPOCHS = config["training"]["epochs"]
PATIENCE = config["training"]["patience"]


LOG_FILE = PROJECT_ROOT / "experiments" / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w")
    ]
)
logger = logging.getLogger("Retrain")

# ────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
# ────────────────────────────────────────────────────────────


def regex_tokenize(text: str) -> list[str]:
    """Better tokenization for source code: splits on punctuation/symbols."""
    return re.findall(r"[\w']+|[^\w\s]", text)


def prepare_data():
    all_texts, all_labels = [], []

    logger.info("📦 Loading datasets...")
    for mal_dir, norm_dir in DATASETS:
        if mal_dir.exists() and norm_dir.exists():
            texts, current_labels = load_multiple_datasets(str(mal_dir), str(norm_dir))
            all_texts.extend(texts)
            all_labels.extend(current_labels)
            logger.info(f"   ✓ Loaded {len(texts)} samples from {mal_dir.parent.name}")
        else:
            logger.warning(f"   ⚠ skipping missing directory: {mal_dir.parent}")

    if not all_texts:
        raise ValueError("No data found! Check your data/ paths.")

    # Tokenizer v2 (now with regex)
    logger.info("🔡 Tokenizing...")
    all_tokens = []
    for text in all_texts:
        all_tokens.extend(regex_tokenize(text))
    
    unique_tokens = sorted(list(set(all_tokens)))
    tokenizer = {word: i + 1 for i, word in enumerate(unique_tokens)}
    sequences = [[tokenizer.get(w, 0) for w in regex_tokenize(text)] for text in all_texts]

    # Pad/Truncate
    logger.info(f"📏 Padding sequences (max_len={MAX_LEN})...")
    padded = np.zeros((len(sequences), MAX_LEN), dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            truncated = seq[:MAX_LEN]
            padded[i, : len(truncated)] = truncated

    return all_texts, np.array(all_labels), tokenizer, padded


# ────────────────────────────────────────────────────────────
# Training Logic with Early Stopping
# ────────────────────────────────────────────────────────────


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(name, model_class, train_loader, val_loader, vocab_size, emb_matrix, device):
    logger.info(f"\n🚀 Training {name}...")
    model = model_class(vocab_size, emb_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    early_stopping = EarlyStopping(patience=PATIENCE)

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item()
                preds.extend((out > 0.5).float().cpu().numpy())
                targets.extend(y.cpu().numpy())

        avg_v_loss = val_loss / len(val_loader)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, zero_division=0)

        logger.info(f"   Epoch {epoch+1:02d}: Loss={total_loss/len(train_loader):.4f}, ValLoss={avg_v_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_path = OUTPUT_DIR / f"{name}_v2.pth"
            torch.save(model.state_dict(), str(save_path))

        early_stopping(avg_v_loss)
        if early_stopping.early_stop:
            logger.info(f"   🛑 Early stopping at epoch {epoch+1:02d}")
            break

    # Re-evaluate the best model to get all MLOps metrics (including ROC-AUC and CM)
    model.load_state_dict(torch.load(str(OUTPUT_DIR / f"{name}_v2.pth"), weights_only=True))
    model.eval()

    all_targets, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            all_probs.extend(out.cpu().numpy().tolist())
            all_preds.extend((out > 0.5).float().cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    from src.evaluation.metrics import evaluate_classification
    metrics = evaluate_classification(all_targets, all_preds, y_probs=all_probs)

    return {
        "Accuracy": metrics.get("accuracy", 0.0),
        "Precision": metrics.get("precision", 0.0),
        "Recall": metrics.get("recall", 0.0),
        "F1-Score": metrics.get("f1_score", 0.0),
        "ROC-AUC": metrics.get("roc_auc", 0.0),
        "Confusion_Matrix": str(metrics.get("confusion_matrix", []))
    }

# ────────────────────────────────────────────────────────────
# Main Execution
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"💻 Device: {device}")

    # 1. Prep Data
    all_texts, labels, tokenizer, padded = prepare_data()
    vocab_size = len(tokenizer) + 1

    # Save Tokenizer v2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info(f"💾 Tokenizer v2 saved to {TOKENIZER_PATH}")

    # 2. Word2Vec
    logger.info("📉 Training Word2Vec embeddings...")
    tokenized_texts = [regex_tokenize(t) for t in all_texts]
    w2v = Word2Vec(tokenized_texts, vector_size=EMBEDDING_DIM, window=5, min_count=1)
    emb_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, idx in tokenizer.items():
        if word in w2v.wv:
            emb_matrix[idx] = w2v.wv[word]

    # 3. Split & Loaders (70% Train, 15% Val, 15% Test)
    logger.info("🔪 Splitting dataset into Train/Val/Test (70/15/15)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        padded, labels, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Train All
    models_to_train = [
        ("WebshellDetector", WebshellDetector),
        ("TextCNN", TextCNN),
        ("TextRNN", TextRNN),
        ("TransformerClassifier", TransformerClassifier),
    ]

    results = {}
    for name, m_class in models_to_train:
        score = train_model(name, m_class, train_loader, val_loader, vocab_size, emb_matrix, device)
        results[name] = score

    logger.info("\n" + "═" * 40)
    logger.info("🏆 RETRAINING COMPLETE")
    logger.info("═" * 40)

    rows = []
    for name, metrics in results.items():
        acc = metrics["Accuracy"]
        prec = metrics["Precision"]
        rec = metrics["Recall"]
        f1 = metrics["F1-Score"]
        logger.info(f" • {name:22} : Acc={acc*100:5.2f}% | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

        # Prepare row for CSV
        metrics["Model"] = name
        rows.append(metrics)

    # Save to CSV
    df = pd.DataFrame(rows)
    # Reorder columns
    df = df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Confusion_Matrix"]]
    csv_path = PROJECT_ROOT / "experiments" / "experiment_results.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"\n📊 Metrics permanently tracked in {csv_path}")
    logger.info(f"⏱ Total time: {(time.time() - start_time)/60:.2f} minutes")
    logger.info(f"📂 Updated models & logs saved in {OUTPUT_DIR}")
