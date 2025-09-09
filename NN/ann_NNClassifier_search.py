import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring

# -------------------
# Select features
# -------------------
selected_features = [
    "all_util", "open_acc_6m", "open_rv_12m", "il_util",
    "delinq_2yrs", "acc_now_delinq", "total_bal_il",
    "last_fico_range_low", "last_fico_range_high", "loan_amnt"
]

df = pd.read_csv('../data/base_dataset_cleaned.csv')
df = df[df['loan_status_mapped'].isin([0, 1])]

X = df[selected_features]
y = df['loan_status_mapped']

# -------------------
# Train/val/test split (70/10/20)
# -------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42
)

# -------------------
# Scale features
# -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -------------------
# Define ANN
# -------------------
class ANNModule(nn.Module):
    def __init__(self, input_dim=10, hidden_layers=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # logits
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).view(-1)

# -------------------
# Wrap with skorch
# -------------------
net = NeuralNetClassifier(
    ANNModule,
    module__input_dim=10,
    criterion=nn.BCEWithLogitsLoss,
    optimizer=optim.Adam,
    max_epochs=20,
    lr=0.001,
    batch_size=1024,
    verbose=1,
    train_split=None,  # we'll pass our own validation set
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Add scoring callbacks for loss tracking
train_loss = EpochScoring(
    scoring='neg_log_loss', name='train_loss', on_train=True
)
valid_loss = EpochScoring(
    scoring='neg_log_loss', name='valid_loss', on_train=False
)

net.set_params(callbacks=[train_loss, valid_loss])

# -------------------
# Hyperparameter grid
# -------------------
params = {
    'module__hidden_layers': [
        [64],
        [128, 64],
        [128, 64, 32]
    ],
    'module__dropout': [0.2, 0.3, 0.5],
    'lr': [0.001, 0.0005],
    'optimizer': [optim.Adam, optim.SGD],
}

# -------------------
# Run Grid Search (with explicit validation set)
# -------------------
gs = GridSearchCV(
    net,
    params,
    refit=True,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=-1
)

gs.fit(
    X_train_scaled.astype(np.float32),
    y_train.values.astype(np.float32),
    X_valid=X_val_scaled.astype(np.float32),
    y_valid=y_val.values.astype(np.float32)
)

print("\nBest params:", gs.best_params_)
print("Best CV Score:", gs.best_score_)

# -------------------
# Save best model
# -------------------
best_model = gs.best_estimator_
torch.save(best_model.module_.state_dict(), "best_ann_model.pth")
print("✅ Model saved as best_ann_model.pth")

# -------------------
# Evaluate on held-out test set
# -------------------
y_pred = gs.predict(X_test_scaled.astype(np.float32))

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred))

# -------------------
# Save CV results
# -------------------
cv_results = pd.DataFrame(gs.cv_results_)
cv_results_subset = cv_results[['params', 'mean_test_score', 'std_test_score']]
cv_results_subset = cv_results_subset.sort_values(by='mean_test_score', ascending=False)
cv_results_subset.to_csv("f1_comparison.csv", index=False)

# -------------------
# Plot F1 comparison
# -------------------
plt.figure(figsize=(12, 6))
plt.errorbar(
    range(len(cv_results_subset)),
    cv_results_subset['mean_test_score'],
    yerr=cv_results_subset['std_test_score'],
    fmt='o',
    capsize=5
)
plt.xticks(
    range(len(cv_results_subset)),
    [str(p) for p in cv_results_subset['params']],
    rotation=90,
    fontsize=8
)
plt.ylabel("Mean F1 Score")
plt.xlabel("Hyperparameter Combination")
plt.title("F1 Score Comparison Across Hyperparameter Combinations")
plt.tight_layout()
plt.savefig("f1_hyperparam_comparison.png")
plt.show()

# -------------------
# Plot Training vs Validation Loss
# -------------------
history = best_model.history
train_losses = [-x for x in history[:, 'train_loss']]
valid_losses = [-x for x in history[:, 'valid_loss']]

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
