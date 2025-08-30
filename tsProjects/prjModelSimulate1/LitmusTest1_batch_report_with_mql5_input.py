
import numpy as np
import pandas as pd
import os
import onnxruntime as rt
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def predict_from_current_price(model_path, recent_closes, sequence_length=5, export_path="mql5_prediction_output.csv"):
    """
    Predict 2h, 8h, and 24h prices from a current price context using ONNX model.
    Inputs:
        model_path (str): path to ONNX model
        recent_closes (list): most recent N close prices, oldest to latest
        sequence_length (int): expected sequence length by model
        export_path (str): output path for MQL5 consumption
    Output:
        Saves CSV with columns: ['predicted_2h', 'predicted_8h', 'predicted_24h']
    """
    if len(recent_closes) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} prices, got {len(recent_closes)}")

    try:
        sess = rt.InferenceSession(model_path, providers=rt.get_available_providers())
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")

    input_sequence = np.array(recent_closes[-sequence_length:], dtype=np.float32).reshape(1, sequence_length, 1)

    try:
        pred = sess.run([output_name], {input_name: input_sequence})[0].item()
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    # In basic setup, output is a single predicted price; we can optionally reuse this for all horizons or expand
    output_df = pd.DataFrame([{
        "predicted_2h": pred,
        "predicted_8h": pred,
        "predicted_24h": pred
    }])
    output_df.to_csv(export_path, index=False)
    print(f"✅ MQL5 prediction exported to {export_path}")


if __name__ == "__main__":
    # Example usage:
    # predict_from_current_price("model.onnx", [1.105, 1.107, 1.106, 1.109, 1.108])

    pass
