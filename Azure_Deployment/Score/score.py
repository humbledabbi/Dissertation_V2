import os
import torch
import torch.nn as nn
import joblib
import numpy as np
import json

# Features in the same order as training
FEATURE_ORDER = [
    "all_util", "open_acc_6m", "open_rv_12m", "il_util",
    "delinq_2yrs", "acc_now_delinq", "total_bal_il",
    "last_fico_range_low", "last_fico_range_high", "loan_amnt"
]

class ANNModule(nn.Module):
    def __init__(self, input_dim=10, hidden_layers=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).view(-1)


def init():
    global model, scaler, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.environ.get("AZUREML_MODEL_DIR", "./model")
    model_path = os.path.join(model_dir, "model", "best_ann_model.pth")
    scaler_path = os.path.join(model_dir, "model", "scaler.pkl")

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load model
    model = ANNModule()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


def run(raw_data):
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # If Azure sends raw JSON string, parse it
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        logger.info(f"Received data: {raw_data}")

        # If it's a dict with "data" key, extract
        if isinstance(raw_data, dict) and "data" in raw_data:
            raw_data = raw_data["data"]

        # Now raw_data should be a list of dicts
        data_list = []
        for d in raw_data:
            if not isinstance(d, dict):
                raise ValueError(f"Expected dict, got {type(d)}: {d}")
            row = [d.get(feat, None) for feat in FEATURE_ORDER]
            if None in row:
                missing = [FEATURE_ORDER[i] for i, v in enumerate(row) if v is None]
                raise ValueError(f"Missing features: {missing}")
            data_list.append(row)

        # Convert to numpy array, scale, and run model
        data = np.array(data_list, dtype=np.float32)
        data_scaled = scaler.transform(data)
        inputs = torch.tensor(data_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

        logger.info(f"Predictions: {preds.cpu().numpy()}")
        return {
            "predictions": preds.cpu().numpy().tolist(),
            "probabilities": probs.cpu().numpy().tolist()
        }

    except Exception as e:
        logger.error(f"Exception in run: {type(e)}: {e}", exc_info=True)
        return {"error": f"{type(e)}: {e}"}
