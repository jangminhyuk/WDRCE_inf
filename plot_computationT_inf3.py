#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file plots the results from the backward time experiment.
It loads the summary data saved by the backward time experiment script and
creates a plot with the simulation horizon (T) on the x–axis and the average
backward computation time (with error bars) on the y–axis for:
   - DRCE (backward time)
   - inf_DRCE (backward time)
   - DRLQC (backward time)
The resulting plot is saved as a PDF file in the folder "./results/backward_time_experiment_lambda/".
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

def main(dist, noise_dist):
    base_path = "./results/backward_time_experiment_lambda/"
    # Updated file name to match the data-generation code.
    data_path = os.path.join(base_path, f"backward_time_costs_exp_time_5_{dist}_{noise_dist}.pkl")
    if not os.path.exists(data_path):
        print("Data file not found:", data_path)
        return
    results = load_data(data_path)
    T_values = np.array(results["T"])
    
    DRCE_mean = np.array(results["DRCE"]["mean"])
    DRCE_std = np.array(results["DRCE"]["std"])
    inf_DRCE_mean = np.array(results["inf_DRCE"]["mean"])
    inf_DRCE_std = np.array(results["inf_DRCE"]["std"])
    DRLQC_mean = np.array(results["DRLQC"]["mean"])
    DRLQC_std = np.array(results["DRLQC"]["std"])
    
    plt.figure(figsize=(8,6))
    plt.errorbar(T_values, DRCE_mean, yerr=DRCE_std, fmt='-o', label='DRCE (finite)')
    plt.errorbar(T_values, inf_DRCE_mean, yerr=inf_DRCE_std, fmt='-s', label='DRCE (infinite)')
    plt.errorbar(T_values, DRLQC_mean, yerr=DRLQC_std, fmt='-^', label='DRLQC (finite)')
    
    plt.xlabel("Time Horizon (T)", fontsize=14)
    plt.ylabel("Offline Computation Time (s)", fontsize=14)
    #plt.title(f"Backward Computation Time vs Time Horizon\n(dist={dist}, noise_dist={noise_dist})", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"backward_time_plot_exp_time_5_{dist}_{noise_dist}.pdf"), dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str,
                        help="Disturbance distribution")
    parser.add_argument('--noise_dist', required=False, default="normal", type=str,
                        help="Noise distribution")
    args = parser.parse_args()
    main(args.dist, args.noise_dist)
