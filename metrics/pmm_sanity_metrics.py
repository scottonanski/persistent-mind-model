#!/usr/bin/env python3
"""PMM Sanity Metrics - Ultra-fast spot check for A/B test results"""
import json
import argparse
import glob
import os


def load_consolidated_json(json_path):
    """Load consolidated A/B test results"""
    with open(json_path, "r") as f:
        return json.load(f)


def count_reflection_attempts_from_logs(json_path):
    """Count reflection attempts from session log files"""
    log_dir = os.path.dirname(json_path)
    session_logs = glob.glob(os.path.join(log_dir, "session_*.log"))

    reflection_counts = {}
    for log_file in session_logs:
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract session_id and condition from filename
            filename = os.path.basename(log_file)
            if "bandit_on" in filename:
                condition = "bandit"
            elif "baseline" in filename:
                condition = "baseline"
            else:
                condition = "unknown"

            # Count reflection attempts in this session
            attempts = content.count("reflection_attempt")

            if condition not in reflection_counts:
                reflection_counts[condition] = []
            reflection_counts[condition].append(attempts)

        except Exception:
            continue  # Skip files we can't read

    return reflection_counts


def extract_metrics(data, json_path):
    """Extract IAS/GAS/Close metrics per condition"""
    conditions = {}

    # Handle ab_test_bandit.py JSON structure
    sessions = data.get("raw_results", [])
    if not sessions:
        # Fallback to other possible structures
        sessions = data.get("sessions", [])
        if not sessions and isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    sessions = value
                    break

    for session in sessions:
        session_id = session.get("session_id", "unknown")

        # Determine condition from session_id
        if "bandit_on" in session_id:
            condition = "bandit"
        elif "baseline" in session_id:
            condition = "baseline"
        else:
            condition = session.get("condition", "unknown")

        if condition not in conditions:
            conditions[condition] = {"ias": [], "gas": [], "close": []}

        # Extract from telemetry arrays (use final values)
        telemetry = session.get("telemetry", {})
        ias_scores = telemetry.get("ias_scores", [0])
        gas_scores = telemetry.get("gas_scores", [0])
        close_rates = telemetry.get("close_rates", [0])

        final_ias = ias_scores[-1] if ias_scores else 0
        final_gas = gas_scores[-1] if gas_scores else 0
        final_close = close_rates[-1] if close_rates else 0

        conditions[condition]["ias"].append(final_ias)
        conditions[condition]["gas"].append(final_gas)
        conditions[condition]["close"].append(final_close)

    # Add reflection attempt counts from event logs
    reflection_counts = count_reflection_attempts_from_logs(json_path)
    for condition in conditions:
        if condition in reflection_counts:
            conditions[condition]["reflection_attempts"] = reflection_counts[condition]
        else:
            conditions[condition]["reflection_attempts"] = []

    return conditions


def safe_pct_change(a, b):
    """Safe percentage change calculation"""
    return ((a - b) / b * 100) if b > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="PMM Sanity Metrics - Quick A/B test readout"
    )
    parser.add_argument(
        "--json",
        default="logs/ab_test_results_bandit_ab_test/ab_test_complete.json",
        help="Path to consolidated JSON results",
    )
    args = parser.parse_args()

    # Load data
    data = load_consolidated_json(args.json)
    conditions = extract_metrics(data, args.json)

    # Print table header
    print(
        f"{'Condition':<15} {'N':<3} {'IAS(avg)':<10} {'GAS(avg)':<10} {'Close(avg)':<10} {'Reflect(avg)':<12}"
    )
    print("-" * 62)

    # Print per-condition metrics
    condition_metrics = {}
    for condition, metrics in conditions.items():
        n = len(metrics["ias"])
        ias_avg = sum(metrics["ias"]) / n if n > 0 else 0
        gas_avg = sum(metrics["gas"]) / n if n > 0 else 0
        close_avg = sum(metrics["close"]) / n if n > 0 else 0

        # Calculate reflection attempts per session
        reflection_attempts = metrics.get("reflection_attempts", [])
        reflect_avg = (
            sum(reflection_attempts) / len(reflection_attempts)
            if reflection_attempts
            else 0
        )

        condition_metrics[condition] = {
            "ias": ias_avg,
            "gas": gas_avg,
            "close": close_avg,
            "reflect": reflect_avg,
            "n": n,
        }
        print(
            f"{condition:<15} {n:<3} {ias_avg:<10.3f} {gas_avg:<10.3f} {close_avg:<10.3f} {reflect_avg:<12.1f}"
        )

    # Calculate deltas vs baseline
    if "baseline" in condition_metrics and "bandit" in condition_metrics:
        baseline = condition_metrics["baseline"]
        bandit = condition_metrics["bandit"]

        print("\nDeltas vs baseline (%, guarded for zeros)")
        ias_delta = safe_pct_change(bandit["ias"], baseline["ias"])
        gas_delta = safe_pct_change(bandit["gas"], baseline["gas"])
        close_delta = safe_pct_change(bandit["close"], baseline["close"])
        reflect_delta = safe_pct_change(bandit["reflect"], baseline["reflect"])

        print(
            f"IAS: {ias_delta:+.1f}%, GAS: {gas_delta:+.1f}%, Close: {close_delta:+.1f}%, Reflect: {reflect_delta:+.1f}%"
        )

        # Recommendation gate
        sample_size = min(baseline["n"], bandit["n"])
        if sample_size < 3:
            print(
                f"\n⚠️  Small sample size (N={sample_size}): Additional validation recommended"
            )
        elif abs(ias_delta) < 10 and abs(gas_delta) < 10 and close_delta > -10:
            print("\n🚀 System ready for production deployment")
        else:
            print("\n⚠️  Consider additional parameter tuning")


if __name__ == "__main__":
    main()
