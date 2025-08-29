#!/usr/bin/env python3
"""PMM Sanity Metrics - Ultra-fast spot check for A/B test results"""
import json
import argparse
from pathlib import Path

def load_consolidated_json(json_path):
    """Load consolidated A/B test results"""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_metrics(data):
    """Extract IAS/GAS/Close metrics per condition"""
    conditions = {}
    
    # Handle both direct sessions list and nested structure
    sessions = data.get('sessions', [])
    if not sessions and isinstance(data, dict):
        # Try alternative structures
        for key, value in data.items():
            if isinstance(value, list):
                sessions = value
                break
    
    for session in sessions:
        condition = session.get('condition', 'unknown')
        if condition not in conditions:
            conditions[condition] = {'ias': [], 'gas': [], 'close': []}
        
        final_metrics = session.get('final_metrics', {})
        conditions[condition]['ias'].append(final_metrics.get('final_IAS', 0))
        conditions[condition]['gas'].append(final_metrics.get('final_GAS', 0))
        conditions[condition]['close'].append(final_metrics.get('final_close_rate', 0))
    
    return conditions

def safe_pct_change(a, b):
    """Safe percentage change calculation"""
    return ((a - b) / b * 100) if b > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description='PMM Sanity Metrics - Quick A/B test readout')
    parser.add_argument('--json', default='logs/ab_test_results_bandit_ab_test/ab_test_complete.json', help='Path to consolidated JSON results')
    args = parser.parse_args()
    
    # Load data
    data = load_consolidated_json(args.json)
    conditions = extract_metrics(data)
    
    # Print table header
    print(f"{'Condition':<15} {'N':<3} {'IAS(avg)':<10} {'GAS(avg)':<10} {'Close(avg)':<10}")
    print("-" * 50)
    
    # Print per-condition metrics
    condition_metrics = {}
    for condition, metrics in conditions.items():
        n = len(metrics['ias'])
        ias_avg = sum(metrics['ias']) / n if n > 0 else 0
        gas_avg = sum(metrics['gas']) / n if n > 0 else 0
        close_avg = sum(metrics['close']) / n if n > 0 else 0
        
        condition_metrics[condition] = {'ias': ias_avg, 'gas': gas_avg, 'close': close_avg, 'n': n}
        print(f"{condition:<15} {n:<3} {ias_avg:<10.3f} {gas_avg:<10.3f} {close_avg:<10.3f}")
    
    # Calculate deltas vs baseline
    if 'baseline' in condition_metrics and 'bandit' in condition_metrics:
        baseline = condition_metrics['baseline']
        bandit = condition_metrics['bandit']
        
        print("\nDeltas vs baseline (%, guarded for zeros)")
        ias_delta = safe_pct_change(bandit['ias'], baseline['ias'])
        gas_delta = safe_pct_change(bandit['gas'], baseline['gas'])
        close_delta = safe_pct_change(bandit['close'], baseline['close'])
        
        print(f"IAS: {ias_delta:+.1f}%, GAS: {gas_delta:+.1f}%, Close: {close_delta:+.1f}%")
        
        # Recommendation gate
        sample_size = min(baseline['n'], bandit['n'])
        if sample_size < 3:
            print(f"\n⚠️  Small sample size (N={sample_size}): Additional validation recommended")
        elif abs(ias_delta) < 10 and abs(gas_delta) < 10 and close_delta > -10:
            print("\n🚀 System ready for production deployment")
        else:
            print("\n⚠️  Consider additional parameter tuning")

if __name__ == "__main__":
    main()
