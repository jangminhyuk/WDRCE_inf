#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file plots the results from the time-horizon experiment (experiment 5).
It loads the summary data saved by the time-horizon experiment script and
creates a plot with the simulation horizon (T) on the x–axis and the time–averaged
control cost (total cost divided by T, with error bars) on the y–axis for:
   - Finite–horizon LQG,
   - Infinite–horizon LQG,
   - Finite–horizon DRCE, 
   - Infinite–horizon DRCE, and 
   - Finite–horizon DRLQC.
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
    # Update folder names to reflect experiment 5.
    if use_lambda_flag:
        base_path = "./results/time_horizon_experiment_5_lambda/"
    else:
        base_path = "./results/time_horizon_experiment_5/"
    # Note the file name now includes "exp5"
    data_path = os.path.join(base_path, f"time_horizon_costs_exp5_{dist}_{noise_dist}.pkl")
    if not os.path.exists(data_path):
        print("Data file not found:", data_path)
        return
    results = load_data(data_path)
    T_values = np.array(results["T"])
    
    # Compute time-averaged cost per step (total cost divided by T).
    LQG_finite_mean   = np.array(results["LQG_finite"]["mean"])   / T_values
    LQG_finite_std    = np.array(results["LQG_finite"]["std"])    / T_values
    LQG_infinite_mean = np.array(results["LQG_infinite"]["mean"]) / T_values
    LQG_infinite_std  = np.array(results["LQG_infinite"]["std"])  / T_values
    DRCE_finite_mean   = np.array(results["DRCE_finite"]["mean"])   / T_values
    DRCE_finite_std    = np.array(results["DRCE_finite"]["std"])    / T_values
    DRCE_infinite_mean = np.array(results["DRCE_infinite"]["mean"]) / T_values
    DRCE_infinite_std  = np.array(results["DRCE_infinite"]["std"])  / T_values
    DRLQC_finite_mean   = np.array(results["DRLQC_finite"]["mean"])   / T_values
    DRLQC_finite_std    = np.array(results["DRLQC_finite"]["std"])    / T_values
    
    plt.figure(figsize=(8,6))
    plt.plot(T_values, LQG_finite_mean, '-o', label='LQG (finite)')
    plt.plot(T_values, LQG_infinite_mean, '-d', label='LQG (infinite)')
    plt.plot(T_values, DRCE_finite_mean, '-s', label='DRCE (finite)')
    plt.plot(T_values, DRCE_infinite_mean, '-^', label='DRCE (infinite)')
    plt.plot(T_values, DRLQC_finite_mean, '-*', label='DRLQC (finite)')
    # Alternatively, to add error bars, you can uncomment the following:
    # plt.errorbar(T_values, LQG_finite_mean, yerr=LQG_finite_std, fmt='-o', label='LQG (finite)')
    # plt.errorbar(T_values, LQG_infinite_mean, yerr=LQG_infinite_std, fmt='-d', label='LQG (infinite)')
    # plt.errorbar(T_values, DRCE_finite_mean, yerr=DRCE_finite_std, fmt='-s', label='DRCE (finite)')
    # plt.errorbar(T_values, DRCE_infinite_mean, yerr=DRCE_infinite_std, fmt='-^', label='DRCE (infinite)')
    # plt.errorbar(T_values, DRLQC_finite_mean, yerr=DRLQC_finite_std, fmt='-*', label='DRLQC (finite)')
    
    plt.xlabel("Time Horizon (T)", fontsize=14)
    plt.ylabel("Time–Averaged Control Cost", fontsize=14)
    plt.title("Time–Averaged Cost vs Time Horizon", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_filename = os.path.join(base_path, f"time_horizon_plot_exp5_{dist}_{noise_dist}.pdf")
    plt.savefig(save_filename, dpi=300)
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
