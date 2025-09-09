import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
)
from imblearn.combine import SMOTEENN

# -------------------
# Config
# -------------------
DATA_PATH = "../data/data_selected.csv"
RESULTS_FILE = "nn_architecture_search_smoteenn_light.csv"
os.makedirs("figures", exist_ok=True)

# -------------------
# Device
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------
# Load & split dataset
# -------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["loan_status_binary", "loan_status_mapped", "installment", "fico_range_high"])
y = df["loan_status_binary"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/val/test split (60/20/20)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

# -------------------
# SMOTEENN function
# -------------------
def apply_smoteenn(X_train, y_train, random_state=42):
    smote_enn = SMOTEENN(random_state=random_state)
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)
    return X_res, y_res

# -------------------
# Convert validation/test sets to tensors
# -------------------
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

# -------------------
# Flexible Neural Network
# -------------------
class FlexibleNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_nodes):
        super(FlexibleNN, self).__init__()
        layers = []
        last_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_nodes))
            layers.append(nn.ReLU())
            last_dim = hidden_nodes
        layers.append(nn.Linear(last_dim, 1))  # output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# -------------------
# Training & Evaluation
# -------------------
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       hidden_layers, hidden_nodes,
                       epochs=100, lr=0.001, device="cpu"):

    # -------------------
    # Apply SMOTEENN
    # -------------------
    X_train_bal, y_train_bal = apply_smoteenn(X_train, y_train.values, random_state=42)

    X_train_tensor = torch.tensor(X_train_bal, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_bal, dtype=torch.float32).unsqueeze(1).to(device)

    # -------------------
    # Model
    # -------------------
    model = FlexibleNN(X_train_tensor.shape[1], hidden_layers, hidden_nodes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0
    best_state = None
    patience, patience_counter = 10, 0

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Validation check
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            precisions, recalls, thresholds = precision_recall_curve(y_val_tensor.cpu().numpy(), val_probs)
            f1s = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precisions, recalls)]
            val_f1 = max(f1s)

        val_losses.append(val_loss)

        # Early stopping check (based on F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # -------------------
    # Test evaluation with threshold sweep
    # -------------------
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    precisions, recalls, thresholds = precision_recall_curve(y_test_tensor.cpu().numpy(), test_probs)
    f1s = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precisions, recalls)]
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (test_probs > best_threshold).astype(int)

    # -------------------
    # Plot training vs validation loss
    # -------------------
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"HL={hidden_layers}, HN={hidden_nodes}, Epochs={epochs}")
    plt.legend()
    plt.grid(True)
    fig_name = f"figures/loss_HL{hidden_layers}_HN{hidden_nodes}_E{epochs}.png"
    plt.savefig(fig_name)
    plt.close()

    return {
        "Hidden Layers": hidden_layers,
        "Hidden Nodes": hidden_nodes,
        "Epochs": epochs,
        "BestThreshold": best_threshold,
        "Accuracy": accuracy_score(y_test_tensor.cpu().numpy(), y_pred),
        "Precision": precision_score(y_test_tensor.cpu().numpy(), y_pred, zero_division=0),
        "Recall": recall_score(y_test_tensor.cpu().numpy(), y_pred, zero_division=0),
        "F1": f1_score(y_test_tensor.cpu().numpy(), y_pred, zero_division=0),
        "Loss Plot": fig_name
    }

# -------------------
# Run experiments
# -------------------
results = []

hidden_layers_list = [3, 4, 5]
hidden_nodes_list = [32, 64, 128]
epoch_list = [100, 150, 200]

for hl in hidden_layers_list:
    for hn in hidden_nodes_list:
        for ep in epoch_list:
            res = train_and_evaluate(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                hidden_layers=hl, hidden_nodes=hn,
                epochs=ep, device=device
            )
            results.append(res)

            print(
                f"HL={hl}, HN={hn}, Epochs={ep} "
                f"--> Acc={res['Accuracy']:.4f}, Prec={res['Precision']:.4f}, "
                f"Rec={res['Recall']:.4f}, F1={res['F1']:.4f}, Thresh={res['BestThreshold']:.4f}"
            )

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(RESULTS_FILE, index=False)
print(f"All results saved to {RESULTS_FILE}")