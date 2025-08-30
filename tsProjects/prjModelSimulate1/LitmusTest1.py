import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as rt
from datetime import datetime, timedelta
import os

ONNX_MODEL_PATH = r"C:\WinRunMnt1\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\Logdir\tsneuromodelid\saved_models\best_model_pytorch.onnx"
MQL5_DATA_PATH = r"C:\WinRunMnt1\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\Mql5Data\EURUSD_ratesdata1_litmus.csv"

def load_ohlc_data_from_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    df['datetime_combined'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['timestamp'].astype(str))
    df.set_index('datetime_combined', inplace=True)
    return df[['open', 'high', 'low', 'close']]

def evaluate_model_backtest(model_path, data_path, sequence_length=5):
    sess = rt.InferenceSession(model_path, providers=rt.get_available_providers())
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    df = load_ohlc_data_from_csv(data_path)
    df = df.sort_index()

    interval_minutes = 1
    step = 60 * 24  # 24h steps
    prediction_offsets = [2*60, 8*60, 24*60]  # 2h, 8h, 24h in minutes

    results = []

    for start_idx in range(0, len(df) - 2 * step - max(prediction_offsets), step):
        past_window = df.iloc[start_idx:start_idx + step]
        future_window = df.iloc[start_idx + step: start_idx + 2 * step]

        if len(past_window) < sequence_length:
            continue

        close_prices = past_window['close'].values
        features = close_prices[-sequence_length:].reshape(1, sequence_length, 1).astype(np.float32)

        try:
            prediction = sess.run([output_name], {input_name: features})[0].item()
        except Exception as e:
            print(f"Prediction failed at index {start_idx}: {e}")
            continue

        actuals = {}
        for offset in prediction_offsets:
            future_time = past_window.index[-1] + timedelta(minutes=offset)
            future_price_row = df.loc[df.index >= future_time]
            if not future_price_row.empty:
                actuals[offset] = future_price_row.iloc[0]['close']
            else:
                actuals[offset] = None

        results.append({
            'input_time': past_window.index[-1],
            'predicted_price': prediction,
            'actual_2h': actuals[120],
            'actual_8h': actuals[480],
            'actual_24h': actuals[1440],
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv("model_backtest_results.csv", index=False)
    print("Backtest results saved to model_backtest_results.csv")
    print(result_df.head())

def visualize_backtest_results(csv_path="model_backtest_results.csv"):
    df = pd.read_csv(csv_path)
    df['error_2h'] = df['predicted_price'] - df['actual_2h']
    df['error_8h'] = df['predicted_price'] - df['actual_8h']
    df['error_24h'] = df['predicted_price'] - df['actual_24h']

    print("Mean Absolute Errors:")
    print("2h:", df['error_2h'].abs().mean())
    print("8h:", df['error_8h'].abs().mean())
    print("24h:", df['error_24h'].abs().mean())

    plt.figure(figsize=(12, 6))
    plt.hist(df['error_2h'].dropna(), bins=50, alpha=0.5, label='2h Error')
    plt.hist(df['error_8h'].dropna(), bins=50, alpha=0.5, label='8h Error')
    plt.hist(df['error_24h'].dropna(), bins=50, alpha=0.5, label='24h Error')
    plt.title('Prediction Errors')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_error_histograms.png")
    plt.show()
    print("Histogram saved as prediction_error_histograms.png")

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    direction_acc = np.mean(np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])) * 100
    return mae, rmse, mape, direction_acc

def visualize_backtest_results(csv_path="model_backtest_results.csv"):
    df = pd.read_csv(csv_path)
    errors = {}
    for h, label in zip([120, 480, 1440], ['2h', '8h', '24h']):
        actual_col = f'actual_{label}'
        df[f'error_{label}'] = df['predicted_price'] - df[actual_col]
        mae, rmse, mape, direction = calculate_metrics(df[actual_col].dropna(), df['predicted_price'][df[actual_col].notna()])
        errors[label] = (mae, rmse, mape, direction)
        print(f"--- {label} ---")
        print(f"MAE: {mae:.5f}, RMSE: {rmse:.5f}, MAPE: {mape:.2f}%, Directional Accuracy: {direction:.2f}%")

    # Plot histogram of errors
    plt.figure(figsize=(12, 6))
    for label in ['2h', '8h', '24h']:
        plt.hist(df[f'error_{label}'].dropna(), bins=50, alpha=0.5, label=f'{label} Error')
    plt.title('Prediction Error Distributions')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_error_histograms.png")
    plt.show()
    print("Histogram saved as prediction_error_histograms.png")

def visualize_backtest_results(csv_path="model_backtest_results.csv"):
    df = pd.read_csv(csv_path)
    results_summary = []

    for h, label in zip([120, 480, 1440], ['2h', '8h', '24h']):
        actual_col = f'actual_{label}'
        error_col = f'error_{label}'
        df[error_col] = df['predicted_price'] - df[actual_col]

        y_true = df[actual_col].dropna()
        y_pred = df['predicted_price'][df[actual_col].notna()]
        mae, rmse, mape, direction = calculate_metrics(y_true, y_pred)

        results_summary.append({
            'horizon': label,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': direction
        })

        print(f"--- {label} ---")
        print(f"MAE: {mae:.5f}, RMSE: {rmse:.5f}, MAPE: {mape:.2f}%, Directional Accuracy: {direction:.2f}%")

    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("model_metrics_summary.csv", index=False)
    print("Metrics summary saved as model_metrics_summary.csv")

    # Plot prediction error histograms
    plt.figure(figsize=(12, 6))
    for label in ['2h', '8h', '24h']:
        plt.hist(df[f'error_{label}'].dropna(), bins=50, alpha=0.5, label=f'{label} Error')
    plt.title('Prediction Error Distributions')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_error_histograms.png")
    plt.show()

    # Plot actual vs predicted price over time (only where all targets exist)
    df_plot = df.dropna(subset=['actual_2h', 'actual_8h', 'actual_24h']).copy()
    df_plot['input_time'] = pd.to_datetime(df_plot['input_time'])

    plt.figure(figsize=(14, 6))
    plt.plot(df_plot['input_time'], df_plot['predicted_price'], label='Predicted', linewidth=2)
    plt.plot(df_plot['input_time'], df_plot['actual_2h'], label='Actual 2h', linestyle='--')
    plt.plot(df_plot['input_time'], df_plot['actual_8h'], label='Actual 8h', linestyle='--')
    plt.plot(df_plot['input_time'], df_plot['actual_24h'], label='Actual 24h', linestyle='--')
    plt.title('Predicted vs Actual Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("predicted_vs_actual_plot.png")
    plt.show()
    print("Line plot saved as predicted_vs_actual_plot.png")

def compare_multiple_models(model_paths, data_path, sequence_length=5):
    df = load_ohlc_data_from_csv(data_path)
    df = df.sort_index()
    interval_minutes = 1
    step = 60 * 24
    prediction_offsets = [2*60, 8*60, 24*60]

    all_results = []

    for model_path in model_paths:
        model_name = Path(model_path).stem
        try:
            sess = rt.InferenceSession(model_path, providers=rt.get_available_providers())
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        for start_idx in range(0, len(df) - 2 * step - max(prediction_offsets), step):
            past_window = df.iloc[start_idx:start_idx + step]

            if len(past_window) < sequence_length:
                continue

            close_prices = past_window['close'].values
            features = close_prices[-sequence_length:].reshape(1, sequence_length, 1).astype(np.float32)

            try:
                prediction = sess.run([output_name], {input_name: features})[0].item()
            except Exception as e:
                print(f"{model_name}: Prediction failed at index {start_idx}: {e}")
                continue

            actuals = {}
            for offset in prediction_offsets:
                future_time = past_window.index[-1] + timedelta(minutes=offset)
                future_price_row = df.loc[df.index >= future_time]
                actuals[offset] = future_price_row.iloc[0]['close'] if not future_price_row.empty else None

            all_results.append({
                'model': model_name,
                'input_time': past_window.index[-1],
                'predicted_price': prediction,
                'actual_2h': actuals[120],
                'actual_8h': actuals[480],
                'actual_24h': actuals[1440],
            })

    pd.DataFrame(all_results).to_csv("model_comparison_results.csv", index=False)
    print("Model comparison results saved to model_comparison_results.csv")

def plot_comparison_leaderboard(csv_path="model_comparison_results.csv"):
    df = pd.read_csv(csv_path)
    summary = []

    for model in df['model'].unique():
        df_model = df[df['model'] == model]
        row = {'model': model}
        for h, label in zip([120, 480, 1440], ['2h', '8h', '24h']):
            actual = df_model[f'actual_{label}'].dropna()
            pred = df_model['predicted_price'][df_model[f'actual_{label}'].notna()]
            mae, rmse, mape, direction = calculate_metrics(actual, pred)
            row[f'{label}_MAE'] = mae
            row[f'{label}_RMSE'] = rmse
            row[f'{label}_MAPE'] = mape
            row[f'{label}_DirectionAcc'] = direction
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("model_comparison_leaderboard.csv", index=False)
    print("Leaderboard saved to model_comparison_leaderboard.csv")
    print(summary_df.sort_values('2h_MAE'))

    # Optional: plot 2h MAE leaderboard
    plt.figure(figsize=(10, 5))
    plt.bar(summary_df['model'], summary_df['2h_MAE'], label='2h MAE')
    plt.title('Model Leaderboard (2h MAE)')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_leaderboard_2h_mae.png")
    plt.show()
    print("Leaderboard plot saved as model_leaderboard_2h_mae.png")

def plot_per_model_lines(csv_path="model_comparison_results.csv"):
    df = pd.read_csv(csv_path)
    df['input_time'] = pd.to_datetime(df['input_time'])

    for model in df['model'].unique():
        df_model = df[df['model'] == model].dropna(subset=['actual_2h'])

        plt.figure(figsize=(14, 6))
        plt.plot(df_model['input_time'], df_model['predicted_price'], label='Predicted', linewidth=2)
        plt.plot(df_model['input_time'], df_model['actual_2h'], label='Actual 2h', linestyle='--')
        plt.title(f'Predicted vs Actual (2h) - {model}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fname = f"predicted_vs_actual_{model}_2h.png".replace('.', '_')
        plt.savefig(fname)
        plt.show()
        print(f"Saved plot: {fname}")

def plot_per_model_lines(csv_path="model_comparison_results.csv", start_date=None, end_date=None):
    df = pd.read_csv(csv_path)
    df['input_time'] = pd.to_datetime(df['input_time'])

    if start_date:
        df = df[df['input_time'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['input_time'] <= pd.to_datetime(end_date)]

    for model in df['model'].unique():
        df_model = df[df['model'] == model].dropna(subset=['actual_2h', 'actual_8h', 'actual_24h'])

        for label in ['2h', '8h', '24h']:
            plt.figure(figsize=(14, 6))
            plt.plot(df_model['input_time'], df_model['predicted_price'], label='Predicted', linewidth=2)
            plt.plot(df_model['input_time'], df_model[f'actual_{label}'], label=f'Actual {label}', linestyle='--')
            plt.title(f'Predicted vs Actual ({label}) - {model}')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fname = f"predicted_vs_actual_{model}_{label}.png".replace('.', '_')
            plt.savefig(fname)
            plt.show()
            print(f"Saved plot: {fname}")

    print("Filtered model comparison saved to model_comparison_results.csv")

def batch_compare_models(model_dir, data_path, start_date=None, end_date=None, output_prefix="batch"):
    import glob
    model_paths = glob.glob(os.path.join(model_dir, "*.onnx"))

    print(f"Found {len(model_paths)} ONNX models to compare in {model_dir}")
    filter_and_compare_models(model_paths, data_path, start_date=start_date, end_date=end_date)

    # Compute metrics for leaderboard
    plot_comparison_leaderboard()

    # Optional: create HTML report from leaderboard CSV
    leaderboard_csv = "model_comparison_leaderboard.csv"
    leaderboard_html = f"{output_prefix}_leaderboard.html"
    try:
        leaderboard_df = pd.read_csv(leaderboard_csv)
        leaderboard_df.to_html(leaderboard_html, index=False)
        print(f"HTML report saved as {leaderboard_html}")
    except Exception as e:
        print(f"Failed to generate HTML report: {e}")

def export_excel_leaderboard(csv_path="model_comparison_leaderboard.csv", top_n=10, output_excel="leaderboard_summary.xlsx"):
    df = pd.read_csv(csv_path)

    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Full Leaderboard")

        for label in ['2h', '8h', '24h']:
            sorted_df = df.sort_values(f'{label}_MAE')
            sorted_df.head(top_n).to_excel(writer, index=False, sheet_name=f"Top {top_n} {label}")

    print(f"Excel summary written to {output_excel}")

import zipfile

def export_excel_leaderboard(csv_path="model_comparison_leaderboard.csv", top_n=10, output_excel="leaderboard_summary.xlsx"):
    df = pd.read_csv(csv_path)

    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Full Leaderboard")

        for label in ['2h', '8h', '24h']:
            sorted_df = df.sort_values(f'{label}_MAE')
            sorted_df.head(top_n).to_excel(writer, index=False, sheet_name=f"Top {top_n} {label}")

        # Add conditional formatting
        workbook = writer.book
        for label in ['2h', '8h', '24h']:
            sheet = writer.sheets[f"Top {top_n} {label}"]
            col_letter = chr(66)  # Column B = MAE
            sheet.conditional_format(f"{col_letter}2:{col_letter}{top_n+1}", {
                "type": "3_color_scale",
                "min_color": "#63BE7B",
                "mid_color": "#FFEB84",
                "max_color": "#F8696B"
            })

    print(f"Excel leaderboard with formatting saved as {output_excel}")

def archive_report_files(zip_name="model_report_bundle.zip"):
    files_to_include = [
        "model_comparison_leaderboard.csv",
        "leaderboard_summary.xlsx",
        "model_comparison_results.csv",
        "model_leaderboard_2h_mae.png",
    ]

    # Include any predicted_vs_actual_*.png plots
    import glob
    files_to_include.extend(glob.glob("predicted_vs_actual_*.png"))

    with zipfile.ZipFile(zip_name, "w") as zipf:
        for file in files_to_include:
            if Path(file).exists():
                zipf.write(file, arcname=Path(file).name)

    print(f"Report ZIP saved as {zip_name}")

if __name__ == "__main__":
    # 1. Run backtest on a reference model
    evaluate_model_backtest(ONNX_MODEL_PATH, MQL5_DATA_PATH)
    visualize_backtest_results()

    # 2. Run batch comparison over folder
    # batch_compare_models("C:/path/to/models", MQL5_DATA_PATH, start_date="2023-01-01", end_date="2023-12-31")

    # 3. Generate Excel with Top-N and color formatting
    # export_excel_leaderboard(top_n=10)

    # 4. ZIP everything (plots, metrics, CSVs)
    # archive_report_files()

from tqdm import tqdm

def filter_and_compare_models(model_paths, data_path, sequence_length=5, start_date=None, end_date=None):
    df = load_ohlc_data_from_csv(data_path)
    df = df.sort_index()

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    interval_minutes = 1
    step = 60 * 24
    prediction_offsets = [2*60, 8*60, 24*60]
    all_results = []

    for model_path in tqdm(model_paths, desc="Evaluating models"):
        model_name = Path(model_path).stem
        try:
            sess = rt.InferenceSession(model_path, providers=rt.get_available_providers())
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        for start_idx in range(0, len(df) - 2 * step - max(prediction_offsets), step):
            past_window = df.iloc[start_idx:start_idx + step]

            if len(past_window) < sequence_length:
                continue

            close_prices = past_window['close'].values
            features = close_prices[-sequence_length:].reshape(1, sequence_length, 1).astype(np.float32)

            try:
                prediction = sess.run([output_name], {input_name: features})[0].item()
            except Exception as e:
                print(f"{model_name}: Prediction failed at index {start_idx}: {e}")
                continue

            actuals = {}
            for offset in prediction_offsets:
                future_time = past_window.index[-1] + timedelta(minutes=offset)
                future_price_row = df.loc[df.index >= future_time]
                actuals[offset] = future_price_row.iloc[0]['close'] if not future_price_row.empty else None

            all_results.append({
                'model': model_name,
                'input_time': past_window.index[-1],
                'predicted_price': prediction,
                'actual_2h': actuals[120],
                'actual_8h': actuals[480],
                'actual_24h': actuals[1440],
            })

    pd.DataFrame(all_results).to_csv("model_comparison_results.csv", index=False)
    
    print("✅ Filtered model comparison complete: model_comparison_results.csv")
