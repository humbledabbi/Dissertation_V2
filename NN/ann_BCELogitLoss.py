import pandas as pd
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv('../data/data_selected.csv')

# Feature-target split
X = df.drop(columns=['loan_status_binary'])
y = df['loan_status_binary']

# First split: 70% train, 30% temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Second split: 1/3 val, 2/3 test → 10% val, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp
)

print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

# Define neural network (no sigmoid here — handled by loss)
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleNN(X_train.shape[1]).to(device)

# Calculate pos_weight for class imbalance
num_pos = (y_train_tensor == 1).sum().item()
num_neg = (y_train_tensor == 0).sum().item()
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
epochs = 20
train_losses, val_losses = [], []

start_time = time.time()
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_tensor)
    loss = criterion(logits, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_loss = criterion(val_logits, y_val_tensor)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if epoch % 2 == 0:
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {loss.item():.4f}, "
              f"Val Loss: {val_loss.item():.4f}")
end_time = time.time()

print(f"\nTraining completed in {end_time - start_time:.2f} seconds.\n")

# --- Plot training vs validation loss ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), train_losses, label="Training Loss")
plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Final evaluation on test set
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).float()

# Metrics
y_pred_cpu = predictions.cpu().numpy()
y_true_cpu = y_test_tensor.cpu().numpy()

print("Classification Report:")
print(classification_report(y_true_cpu, y_pred_cpu, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_true_cpu, y_pred_cpu))

accuracy = accuracy_score(y_true_cpu, y_pred_cpu)
precision = precision_score(y_true_cpu, y_pred_cpu, zero_division=0)
recall = recall_score(y_true_cpu, y_pred_cpu, zero_division=0)
f1 = f1_score(y_true_cpu, y_pred_cpu, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true_cpu, y_pred_cpu).ravel()

# Create a DataFrame for saving
metrics_df = pd.DataFrame([{
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'True Negative': tn,
    'False Positive': fp,
    'False Negative': fn,
    'True Positive': tp,
    'Training Time (s)': round(end_time - start_time, 2),
    'Epochs': epochs,
    'Model': 'PyTorch_NN_with_BCEWithLogits'
}])

# Save to CSV (same name as before)
metrics_df.to_csv("nn_metrics.csv", index=False)
print("\nMetrics saved to nn_metrics.csv")
