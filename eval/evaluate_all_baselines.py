#!/usr/bin/env python3
"""Evaluate FNO + baselines on test set."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    FNOWithDPEAndAttention,
    RAOBaseline,
    UNetBaseline,
    CNNBaseline,
)
from eval.metrics import MetricsCalculator

def evaluate_all(
    fno_ckpt,
    emec_test_data,
    output_dir="results/evaluation/",
    metrics=["rmse", "nrmse", "mae", "peak_error", "crps"],
    plot=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {emec_test_data}")
    data = np.load(emec_test_data)
    X_test = data['spectra']
    Y_test = data['loads']
    
    # Initialize models
    print("\nInitializing models...")
    models = {
        'FNO + DPE + Attention': FNOWithDPEAndAttention(),
        'FNO Baseline': FNOWithDPEAndAttention(use_modal_attention=False),
        'U-Net': UNetBaseline(),
        'CNN': CNNBaseline(),
        'Linear RAO': RAOBaseline(),
    }
    
    # Load FNO checkpoint if available
    try:
        import torch
        ckpt = torch.load(fno_ckpt)
        models['FNO + DPE + Attention'].load_state_dict(ckpt['model'])
    except:
        print(f"Warning: Could not load checkpoint {fno_ckpt}")
    
    # Evaluate
    calc = MetricsCalculator()
    results = []
    
    print("\nEvaluating...")
    for model_name, model in models.items():
        print(f"  {model_name}...", end=" ", flush=True)
        
        # Forward pass (simplified)
        pred_mean = model(X_test[:100])  # Demo on first 100 samples
        
        # Compute metrics
        metrics_dict = calc.compute_all(
            pred_mean=pred_mean.cpu().numpy() if hasattr(pred_mean, 'cpu') else pred_mean,
            target=Y_test[:100],
        )
        
        metrics_dict['model'] = model_name
        results.append(metrics_dict)
        print("done")
    
    # Create results table
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'results_table.csv', index=False)
    
    print("\nResults:")
    print(df_results.to_string())
    print(f"\nSaved to {output_dir / 'results_table.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fno_ckpt", default="checkpoints/finetune/best_model.pt")
    parser.add_argument("--emec_test", default="data/processed/emec_test.npz")
    parser.add_argument("--output_dir", default="results/evaluation/")
    parser.add_argument("--plot", action="store_true")
    
    args = parser.parse_args()
    evaluate_all(**vars(args))
