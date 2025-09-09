import os
import time
import pandas as pd
import cupy as cp
import gc

from cuml.svm import SVC
from cuml.model_selection import train_test_split
from cuml.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load data
data = pd.read_csv('../data/data_selected.csv')
X = data.drop(columns=['loan_status_mapped', 'loan_status_binary'])
y = data['loan_status_binary']

# Move to GPU
X_gpu = cp.asarray(X.values, dtype=cp.float32)
y_gpu = cp.asarray(y.values, dtype=cp.int32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_gpu, y_gpu, test_size=0.2, random_state=112, stratify=y_gpu
)

print(f'How much are we running {len(X_train)}')

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

output_dir = 'models_cuMLsvm'
os.makedirs(output_dir, exist_ok=True)

param_grid = [
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'linear', 'C': 10},
    {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 1, 'gamma': 'auto'},
    {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
    {'kernel': 'poly', 'C': 1, 'degree': 2, 'gamma': 'scale'},
    {'kernel': 'poly', 'C': 1, 'degree': 3, 'gamma': 'scale'},
    {'kernel': 'poly', 'C': 10, 'degree': 2, 'gamma': 'auto'},
    {'kernel': 'poly', 'C': 10, 'degree': 3, 'gamma': 'auto'}
]

results = []

for params in param_grid:
    model_name = "_".join([f"{k}={v}" for k, v in params.items()])
    print(f"Training model: {model_name}")

    try:
        clf = SVC(cache_size=4096, **params)
    
        # Train
        start = time.time()
        clf.fit(X_train_scaled, y_train)
        train_time = time.time() - start
    
        # Predict
        y_pred = clf.predict(X_test_scaled)
        y_pred_cpu = cp.asnumpy(y_pred)
        y_test_cpu = cp.asnumpy(y_test)
    
        # Evaluate
        acc = accuracy_score(y_test_cpu, y_pred_cpu)
        err = 1 - acc
        tn, fp, fn, tp = confusion_matrix(y_test_cpu, y_pred_cpu).ravel()
    
        # Save model
        model_file = os.path.join(output_dir, f"{model_name}.pkl")
        joblib.dump(clf, model_file)
    
        # Save result
        results.append({
            **params,
            'accuracy': acc,
            'error_rate': err,
            'train_time_sec': train_time,
            'model_file': model_file,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp
        })
    except Exception as e:
        print(f"Failed: {model_name} — {e}")
    finally:
        if clf is not None:
            del clf
        cp._default_memory_pool.free_all_blocks()
        gc.collect()

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)

print("Models saved")
