import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

# -------------------
# Config
# -------------------
DATA_PATH = "../data/data_selected.csv"
RESULTS_FILE = "nn_architecture_search_undersample.csv"
os.makedirs("figures", exist_ok=True)

# -------------------
# Device
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------
# Load & split dataset (70/10/20)
# -------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["loan_status_binary", "loan_status_mapped", "installment", "fico_range_high"])
y = df["loan_status_binary"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Second split: from temp → 2/3 test (20%), 1/3 validation (10%)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
)

# -------------------
# Undersampling function
# -------------------
def undersample(X, y, ratio=1.0, random_state=None):
    np.random.seed(random_state)
    
    X_pos = X[y == 1]
    y_pos = y[y == 1]
    
    X_neg = X[y == 0]
    y_neg = y[y == 0]

    n_pos = len(y_pos)
    n_neg_sample = int(n_pos * ratio)
    
    idx = np.random.choice(len(y_neg), size=n_neg_sample, replace=False)
    
    X_neg_sampled = X_neg[idx] if isinstance(X_neg, np.ndarray) else X_neg.iloc[idx].to_numpy()
    y_neg_sampled = y_neg.iloc[idx]

    X_bal = np.vstack([X_pos, X_neg_sampled])
    y_bal = np.hstack([y_pos, y_neg_sampled])

    # Shuffle
    perm = np.random.permutation(len(y_bal))
    X_bal = X_bal[perm]
    y_bal = y_bal[perm]
    
    return X_bal, y_bal

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
    # Undersample training data
    # -------------------
    X_train_bal, y_train_bal = undersample(X_train, y_train, ratio=1.0, random_state=42)
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

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Validation F1
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            precisions, recalls, thresholds = precision_recall_curve(y_val_tensor.cpu().numpy(), val_probs)
            f1s = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precisions, recalls)]
            val_f1 = max(f1s)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
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

    return {
        "Hidden Layers": hidden_layers,
        "Hidden Nodes": hidden_nodes,
        "Epochs": epochs,
        "BestThreshold": best_threshold,
        "Accuracy": accuracy_score(y_test_tensor.cpu().numpy(), y_pred),
        "Precision": precision_score(y_test_tensor.cpu().numpy(), y_pred, zero_division=0),
        "Recall": recall_score(y_test_tensor.cpu().numpy(), y_pred, zero_division=0),
        "F1": f1_score(y_test_tensor.cpu().numpy(), y_pred, zero_division=0),
    }

# -------------------
# Run experiments
# -------------------
results = []

hidden_layers_list = [1, 2, 3]
hidden_nodes_list = [32, 64, 128]
epoch_list = [50, 100, 150, 200]

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