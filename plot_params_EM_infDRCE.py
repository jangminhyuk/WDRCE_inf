#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file generates a 3D figure comparing the performance of 
LQG, finite–horizon DRCE, and infinite–horizon DRCE.
It assumes that the simulation code saved results as:
   - "lqg.pkl" for LQG,
   - "drce_finite<param>and<theta_v>.pkl" for finite–horizon DRCE, and
   - "drce_infinite<param>and<theta_v>.pkl" for infinite–horizon DRCE.
In lambda mode, the <param> field is the lambda value; in theta mode it is $\theta_w$.
Since we are not using an infinite–horizon LQG controller, only the above three methods are plotted.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import re
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_experiment_lambda(dist, noise_dist):
    # Set base path for lambda experiment results.
    base_path = "./results/{}_{}/experiment/params_lambda/".format(dist, noise_dist)
    # If no files are found in base_path, try the "raw" subdirectory.
    if not os.listdir(base_path):
        base_path = os.path.join(base_path, "raw")
    
    # Regex patterns for lambda mode.
    # Files are assumed to be named as follows:
    #  - finite–horizon DRCE: "drce_finite_<lambda>and_<theta_v>.pkl"
    #  - infinite–horizon DRCE: "drce_infinite_<lambda>and_<theta_v>.pkl"
    #  - LQG: "lqg.pkl"
    pattern_drce_finite = r"drce_finite_(\d+(?:_\d+)?)and_(\d+(?:_\d+)?)\.pkl"
    pattern_drce_infinite = r"drce_infinite_(\d+(?:_\d+)?)and_(\d+(?:_\d+)?)\.pkl"
    pattern_lqg = r"lqg\.pkl"
    
    # Helper: convert underscore string (e.g., "20" or "20_0") to float.
    def convert_to_float(underscore_value):
        return float(underscore_value.replace('_', '.'))
    
    lqg_lambda_values = []
    lqg_theta_v_values = []
    lqg_cost_values = []
    
    drce_lambda_values = []
    drce_theta_v_values = []
    drce_cost_values = []
    
    drce_inf_lambda_values = []
    drce_inf_theta_v_values = []
    drce_inf_cost_values = []
    
    # Loop over files in the base directory.
    for filename in os.listdir(base_path):
        match_finite = re.search(pattern_drce_finite, filename)
        if match_finite:
            lam_val = convert_to_float(match_finite.group(1))
            theta_v_val = convert_to_float(match_finite.group(2))
            with open(os.path.join(base_path, filename), 'rb') as f:
                cost_val = pickle.load(f)
            drce_lambda_values.append(lam_val)
            drce_theta_v_values.append(theta_v_val)
            drce_cost_values.append(cost_val)
            continue
        
        match_infinite = re.search(pattern_drce_infinite, filename)
        if match_infinite:
            lam_val = convert_to_float(match_infinite.group(1))
            theta_v_val = convert_to_float(match_infinite.group(2))
            with open(os.path.join(base_path, filename), 'rb') as f:
                cost_val = pickle.load(f)
            drce_inf_lambda_values.append(lam_val)
            drce_inf_theta_v_values.append(theta_v_val)
            drce_inf_cost_values.append(cost_val)
            continue
        
        match_lqg = re.search(pattern_lqg, filename)
        if match_lqg:
            with open(os.path.join(base_path, filename), 'rb') as f:
                cost_val = pickle.load(f)
            # LQG cost is independent of the ambiguity parameters.
            # We replicate the LQG cost for every combination.
            if dist == "normal":
                lambda_list = [15, 20, 25, 30, 35, 40]
                theta_v_list = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
            else:
                lambda_list = [15, 20, 25, 30, 35, 40]
                theta_v_list = [1.0, 2.0, 3.0, 4.0]
            for lam in lambda_list:
                for tv in theta_v_list:
                    lqg_lambda_values.append(lam)
                    lqg_theta_v_values.append(tv)
                    lqg_cost_values.append(cost_val)
            continue

    # Convert lists to numpy arrays.
    lqg_lambda_values = np.array(lqg_lambda_values)
    lqg_theta_v_values = np.array(lqg_theta_v_values)
    lqg_cost_values = np.array(lqg_cost_values)
    
    drce_lambda_values = np.array(drce_lambda_values)
    drce_theta_v_values = np.array(drce_theta_v_values)
    drce_cost_values = np.array(drce_cost_values)
    
    drce_inf_lambda_values = np.array(drce_inf_lambda_values)
    drce_inf_theta_v_values = np.array(drce_inf_theta_v_values)
    drce_inf_cost_values = np.array(drce_inf_cost_values)
    
    # Create a 3D surface plot.
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # LQG surface.
    lambda_grid_lqg, theta_v_grid_lqg = np.meshgrid(
        np.linspace(np.min(lqg_lambda_values), np.max(lqg_lambda_values), 100),
        np.linspace(np.min(lqg_theta_v_values), np.max(lqg_theta_v_values), 100)
    )
    cost_grid_lqg = griddata(
        (lqg_lambda_values, lqg_theta_v_values), lqg_cost_values,
        (lambda_grid_lqg, theta_v_grid_lqg), method='cubic'
    )
    surface_lqg = ax.plot_surface(lambda_grid_lqg, theta_v_grid_lqg, cost_grid_lqg,
                                  alpha=0.5, color='red')
    
    # Finite-horizon DRCE surface.
    lambda_grid_drce, theta_v_grid_drce = np.meshgrid(
        np.linspace(np.min(drce_lambda_values), np.max(drce_lambda_values), 100),
        np.linspace(np.min(drce_theta_v_values), np.max(drce_theta_v_values), 100)
    )
    cost_grid_drce = griddata(
        (drce_lambda_values, drce_theta_v_values), drce_cost_values,
        (lambda_grid_drce, theta_v_grid_drce), method='cubic'
    )
    surface_drce = ax.plot_surface(lambda_grid_drce, theta_v_grid_drce, cost_grid_drce,
                                   alpha=0.5, color='green')
    
    # Infinite-horizon DRCE surface.
    lambda_grid_inf, theta_v_grid_inf = np.meshgrid(
        np.linspace(np.min(drce_inf_lambda_values), np.max(drce_inf_lambda_values), 100),
        np.linspace(np.min(drce_inf_theta_v_values), np.max(drce_inf_theta_v_values), 100)
    )
    cost_grid_inf = griddata(
        (drce_inf_lambda_values, drce_inf_theta_v_values), drce_inf_cost_values,
        (lambda_grid_inf, theta_v_grid_inf), method='cubic'
    )
    surface_inf = ax.plot_surface(lambda_grid_inf, theta_v_grid_inf, cost_grid_inf,
                                  alpha=0.5, color='blue')
    
    # Create legend.
    surfaces = [surface_lqg, surface_drce, surface_inf]
    labels = ['LQG (infinite)', 'DRCE (finite)', 'DRCE (infinite)']
    legend = fig.legend(surfaces, labels, bbox_to_anchor=(0.98, 0.5),
                        loc='center right', frameon=True, framealpha=1.0,
                        facecolor='white', fontsize=18,
                        borderpad=0.3, handletextpad=0.2, labelspacing=0.1)
    legend.get_frame().set_alpha(0.7)
    legend.get_frame().set_facecolor('white')
    
    ax.set_xlabel(r'$\lambda$', fontsize=24, labelpad=8)
    ax.set_ylabel(r'$\theta_v$', fontsize=24, labelpad=8)
    ax.set_zlabel(r'Total Cost', fontsize=24, rotation=90, labelpad=5)
    
    ax.tick_params(axis='z', which='major', labelsize=18, pad=2)
    ax.tick_params(axis='x', which='major', labelsize=18, pad=0)
    ax.tick_params(axis='y', which='major', labelsize=18, pad=0)
    
    ax.view_init(elev=14, azim=35)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)
    
    plt.show()
    save_path = os.path.join(base_path, 'experiment_{}_{}_comparison.pdf'.format(dist, noise_dist))
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str,
                        help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', required=False, default="normal", type=str,
                        help="Noise distribution (normal or quadratic)")
    parser.add_argument('--infinite', required=False, action="store_true",
                        help="Infinite horizon flag (if flagged)")
    parser.add_argument('--use_lambda', required=False, action="store_true",
                        help="Use lambda-based experiment instead of theta_w")
    args = parser.parse_args()
    
    # For this experiment we assume lambda mode.
    base_path = "./results/{}_{}/experiment/params_lambda/".format(args.dist, args.noise_dist)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if not os.listdir(base_path):
        base_path = os.path.join(base_path, "raw")
    plot_experiment_lambda(args.dist, args.noise_dist)

if __name__ == "__main__":
    main()
