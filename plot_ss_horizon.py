#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file plots the results from the time-horizon experiment.
It loads the summary data saved by generate_time_horizon_experiment.py and
creates a plot with the simulation horizon (T) on the x–axis and the time–averaged
control cost (total cost divided by T, with error bars) on the y–axis for:
   - LQG,
   - Finite–horizon DRCE, and 
   - Infinite–horizon DRCE.
When using lambda (use_lambda=True), the results are read from a different folder.
The plot is saved as a PDF file.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def main(dist, noise_dist, use_lambda_flag):
    if use_lambda_flag:
        base_path = "./results/time_horizon_experiment_lambda/"
    else:
        base_path = "./results/time_horizon_experiment/"
    data_path = os.path.join(base_path, f"time_horizon_costs_{dist}_{noise_dist}.pkl")
    if not os.path.exists(data_path):
        print("Data file not found:", data_path)
        return
    results = load_data(data_path)
    T_values = np.array(results["T"])
    
    # Compute time-averaged cost per step.
    LQG_mean = np.array(results["LQG"]["mean"]) / T_values
    LQG_std = np.array(results["LQG"]["std"]) / T_values
    DRCE_finite_mean = np.array(results["DRCE_finite"]["mean"]) / T_values
    DRCE_finite_std = np.array(results["DRCE_finite"]["std"]) / T_values
    DRCE_infinite_mean = np.array(results["DRCE_infinite"]["mean"]) / T_values
    DRCE_infinite_std = np.array(results["DRCE_infinite"]["std"]) / T_values
    
    plt.figure(figsize=(8,6))
    plt.plot(T_values, LQG_mean, '-o', label='LQG')
    plt.plot(T_values, DRCE_finite_mean, '-s', label='DRCE (finite)')
    plt.plot(T_values, DRCE_infinite_mean, '-^', label='DRCE (infinite)')
    # plt.errorbar(T_values, LQG_mean, yerr=LQG_std, fmt='-o', label='LQG')
    # plt.errorbar(T_values, DRCE_finite_mean, yerr=DRCE_finite_std, fmt='-s', label='DRCE (finite)')
    # plt.errorbar(T_values, DRCE_infinite_mean, yerr=DRCE_infinite_std, fmt='-^', label='DRCE (infinite)')
    plt.xlabel("Time Horizon (T)", fontsize=14)
    plt.ylabel("Time–Averaged Control Cost", fontsize=14)
    plt.title(f"Time–Averaged Cost vs Time Horizon\n(dist={dist}, noise_dist={noise_dist})", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"time_horizon_plot_{dist}_{noise_dist}.pdf"), dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str,
                        help="Disturbance distribution")
    parser.add_argument('--noise_dist', required=False, default="normal", type=str,
                        help="Noise distribution")
    parser.add_argument('--use_lambda', required=False, action="store_true",
                        help="If set, read results from the lambda experiment folder")
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.use_lambda)
