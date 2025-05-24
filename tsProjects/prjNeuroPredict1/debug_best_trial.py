
import requests
import json

ORACLE_HOST = "192.168.1.103"
ORACLE_PORT = 9000

def fetch_trials():
    try:
        url = f"http://{ORACLE_HOST}:{ORACLE_PORT}/list_trials"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        trials = response.json().get("trials", [])
        return trials
    except Exception as e:
        print(f"❌ Error fetching trials: {e}")
        return []

def print_summary(trials):
    if not trials:
        print("⚠️ No trials returned by OracleServer.")
        return

    completed = [t for t in trials if t["status"] == "COMPLETED" and t.get("score") is not None]
    failed = [t for t in trials if t["status"] == "FAILED"]
    running = [t for t in trials if t["status"] == "RUNNING"]

    print(f"\n📊 Trial Summary:")
    print(f"- Total Trials: {len(trials)}")
    print(f"- Completed: {len(completed)}")
    print(f"- Running: {len(running)}")
    print(f"- Failed: {len(failed)}")

    if completed:
        best = sorted(completed, key=lambda x: x["score"])[0]
        print(f"\n🏆 Best Trial: ID {best['trial_id']} with score {best['score']}")
        print("Hyperparameters:")
        for k, v in best.get("hyperparameters", {}).items():
            print(f"  - {k}: {v}")
    else:
        print("❌ No completed trials with a valid score were found.")

def print_failed(trials):
    failed = [t for t in trials if t["status"] == "FAILED"]
    if failed:
        print("\n🛑 Failed Trials:")
        for t in failed:
            print(f"- Trial {t['trial_id']}")

def main():
    trials = fetch_trials()
    print_summary(trials)
    print_failed(trials)

if __name__ == "__main__":
    main()
