import pandas as pd
import numpy as np
import onnxruntime as rt
from datetime import timedelta
import argparse
import os
import matplotlib.pyplot as plt
import zipfile

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    df.set_index('datetime', inplace=True)
    return df

def predict_from_24h_window(df, model_path, sequence_length=5):
    if len(df) < sequence_length:
        raise ValueError("Not enough data in the 24h back window.")
    close_seq = df['close'].values[-sequence_length:]
    input_data = close_seq.reshape(1, sequence_length, 1).astype(np.float32)
    sess = rt.InferenceSession(model_path, providers=rt.get_available_providers())
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred = sess.run([output_name], {input_name: input_data})[0].item()
    return pred

def evaluate_predictions(df_forward, prediction_time, prediction):
    offsets = {
        "2h": prediction_time + timedelta(hours=2),
        "8h": prediction_time + timedelta(hours=8),
        "24h": prediction_time + timedelta(hours=24)
    }
    results = {"prediction_time": prediction_time, "predicted_price": prediction}
    for label, ts in offsets.items():
        next_val = df_forward[df_forward.index >= ts]
        if not next_val.empty:
            actual = next_val.iloc[0]['close']
            error = abs(prediction - actual)
            results[f'actual_{label}'] = actual
            results[f'error_{label}'] = error
        else:
            results[f'actual_{label}'] = None
            results[f'error_{label}'] = None
    return results

def plot_prediction(label, pred, actual, prediction_time, outpath):
    try:
        plt.figure(figsize=(8, 4))
        plt.axhline(y=pred, color='blue', linestyle='-', label='Predicted')
        plt.axhline(y=actual, color='green', linestyle='--', label=f'Actual {label}')
        plt.title(f"{label.upper()} Forecast vs Actual @ {prediction_time.strftime('%Y-%m-%d %H:%M')}")
        plt.xlabel("Time Horizon")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outpath)
        print(f"🖼️ Plot saved: {outpath}")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plotting failed for {label}: {e}")

def zip_outputs(files, zip_path="mql5_prediction_bundle.zip"):
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f in files:
            if os.path.exists(f):
                zipf.write(f, arcname=os.path.basename(f))
    print(f"📦 Results archived to: {zip_path}")

def main(input_csv, model_path, output_csv, sequence_length=5, plot_base=None):
    df = load_data(input_csv)
    midpoint = df.index.min() + timedelta(hours=24)
    df_back = df[df.index < midpoint]
    df_forward = df[df.index >= midpoint]
    prediction_time = df_back.index.max()

    prediction = predict_from_24h_window(df_back, model_path, sequence_length)
    results = evaluate_predictions(df_forward, prediction_time, prediction)

    out_files = [output_csv]
    if plot_base:
        for label in ['2h', '8h', '24h']:
            actual = results.get(f'actual_{label}')
            if actual:
                plot_path = f"{plot_base}_{label}.png"
                plot_prediction(label, prediction, actual, prediction_time, plot_path)
                out_files.append(plot_path)

    pd.DataFrame([results]).to_csv(output_csv, index=False)
    zip_outputs(out_files)
    print(f"✅ Results written to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="48h OHLC input CSV")
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--output", default="mql5_backtest_result.csv", help="CSV output path")
    parser.add_argument("--seq", type=int, default=5, help="Sequence length")
    parser.add_argument("--plot", help="Base path for plot images")

    args = parser.parse_args()
    main(args.input, args.model, args.output, args.seq, args.plot)