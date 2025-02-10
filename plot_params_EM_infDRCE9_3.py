#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file plots a 3D figure comparing the performance of three infinite–horizon controllers:
   - inf_LQG,
   - inf_WDRC, and 
   - inf_DRCE.
It assumes that the simulation code saved results as:
   - "inf_lqg.pkl" for inf_LQG,
   - "inf_wdrc_<lambda>and_<theta_v>.pkl" for inf_WDRC, and
   - "inf_drce_<lambda>and_<theta_v>.pkl" for inf_DRCE.
The results are read from a folder whose name includes "params_lambda3".
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

def plot_experiment_lambda3(dist, noise_dist):
    # Set base path for lambda experiment results (using folder "params_lambda3")
    base_path = "./results/{}_{}/experiment/params_lambda3/".format(dist, noise_dist)
    if not os.path.exists(base_path) or not os.listdir(base_path):
        base_path = os.path.join(base_path, "raw")
    
    # Define regex patterns for the three controllers.
    pattern_inf_drce = r"drce_finite_(\d+(?:_\d+)?)and_(\d+(?:_\d+)?)\.pkl"
    pattern_inf_wdrc = r"drce_infinite_(\d+(?:_\d+)?)and_(\d+(?:_\d+)?)\.pkl"
    pattern_inf_lqg   = r"lqg\.pkl"


    # Helper: convert an underscore string (e.g. "20" or "20_0") to a float.
    def convert_to_float(val):
        return float(val.replace('_', '.'))
    
    # Lists to collect parameter values and cost for each method.
    inf_lqg_lambda_vals = []
    inf_lqg_theta_v_vals = []
    inf_lqg_cost_vals = []
    
    inf_wdrc_lambda_vals = []
    inf_wdrc_theta_v_vals = []
    inf_wdrc_cost_vals = []
    
    inf_drce_lambda_vals = []
    inf_drce_theta_v_vals = []
    inf_drce_cost_vals = []
    
    # Loop over all files in the base directory.
    for filename in os.listdir(base_path):
        # Check for infinite DRCE file.
        match_drce = re.search(pattern_inf_drce, filename)
        if match_drce:
            lam_val = convert_to_float(match_drce.group(1))
            theta_v_val = convert_to_float(match_drce.group(2))
            cost_val = load_data(os.path.join(base_path, filename))
            inf_drce_lambda_vals.append(lam_val)
            inf_drce_theta_v_vals.append(theta_v_val)
            inf_drce_cost_vals.append(cost_val)
            continue
        
        # Check for infinite WDRC file.
        match_wdrc = re.search(pattern_inf_wdrc, filename)
        if match_wdrc:
            lam_val = convert_to_float(match_wdrc.group(1))
            theta_v_val = convert_to_float(match_wdrc.group(2))
            cost_val = load_data(os.path.join(base_path, filename))
            inf_wdrc_lambda_vals.append(lam_val)
            inf_wdrc_theta_v_vals.append(theta_v_val)
            inf_wdrc_cost_vals.append(cost_val)
            continue
        
        # Check for infinite LQG file.
        match_lqg = re.search(pattern_inf_lqg, filename)
        if match_lqg:
            cost_val = load_data(os.path.join(base_path, filename))
            # For inf_LQG, the cost is independent of lambda and theta_v.
            # Define the parameter ranges based on the disturbance type.
            if dist == "normal":
                lambda_list = np.array([30, 40, 50, 60])
                theta_v_list = np.array([0.01, 0.05, 0.1])
            else:
                lambda_list = np.array([15, 20, 25, 30, 35, 40])
                theta_v_list = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            # Replicate the inf_LQG cost for every parameter combination.
            for lam in lambda_list:
                for tv in theta_v_list:
                    inf_lqg_lambda_vals.append(lam)
                    inf_lqg_theta_v_vals.append(tv)
                    inf_lqg_cost_vals.append(cost_val)
            continue
    
    # Convert lists to numpy arrays.
    inf_lqg_lambda_vals = np.array(inf_lqg_lambda_vals)
    inf_lqg_theta_v_vals = np.array(inf_lqg_theta_v_vals)
    inf_lqg_cost_vals = np.array(inf_lqg_cost_vals)
    
    inf_wdrc_lambda_vals = np.array(inf_wdrc_lambda_vals)
    inf_wdrc_theta_v_vals = np.array(inf_wdrc_theta_v_vals)
    inf_wdrc_cost_vals = np.array(inf_wdrc_cost_vals)
    
    inf_drce_lambda_vals = np.array(inf_drce_lambda_vals)
    inf_drce_theta_v_vals = np.array(inf_drce_theta_v_vals)
    inf_drce_cost_vals = np.array(inf_drce_cost_vals)
    
    # Create 3D surfaces using griddata interpolation.
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid for inf_LQG.
    lambda_grid_lqg, theta_v_grid_lqg = np.meshgrid(
        np.linspace(np.min(inf_lqg_lambda_vals), np.max(inf_lqg_lambda_vals), 100),
        np.linspace(np.min(inf_lqg_theta_v_vals), np.max(inf_lqg_theta_v_vals), 100)
    )
    cost_grid_lqg = griddata(
        (inf_lqg_lambda_vals, inf_lqg_theta_v_vals), inf_lqg_cost_vals,
        (lambda_grid_lqg, theta_v_grid_lqg), method='cubic'
    )
    surface_lqg = ax.plot_surface(lambda_grid_lqg, theta_v_grid_lqg, cost_grid_lqg,
                                  alpha=0.5, color='red')
    
    # Create grid for inf_WDRC.
    lambda_grid_wdrc, theta_v_grid_wdrc = np.meshgrid(
        np.linspace(np.min(inf_wdrc_lambda_vals), np.max(inf_wdrc_lambda_vals), 100),
        np.linspace(np.min(inf_wdrc_theta_v_vals), np.max(inf_wdrc_theta_v_vals), 100)
    )
    cost_grid_wdrc = griddata(
        (inf_wdrc_lambda_vals, inf_wdrc_theta_v_vals), inf_wdrc_cost_vals,
        (lambda_grid_wdrc, theta_v_grid_wdrc), method='cubic'
    )
    surface_wdrc = ax.plot_surface(lambda_grid_wdrc, theta_v_grid_wdrc, cost_grid_wdrc,
                                   alpha=0.5, color='blue')
    
    # Create grid for inf_DRCE.
    lambda_grid_drce, theta_v_grid_drce = np.meshgrid(
        np.linspace(np.min(inf_drce_lambda_vals), np.max(inf_drce_lambda_vals), 100),
        np.linspace(np.min(inf_drce_theta_v_vals), np.max(inf_drce_theta_v_vals), 100)
    )
    cost_grid_drce = griddata(
        (inf_drce_lambda_vals, inf_drce_theta_v_vals), inf_drce_cost_vals,
        (lambda_grid_drce, theta_v_grid_drce), method='cubic'
    )
    surface_drce = ax.plot_surface(lambda_grid_drce, theta_v_grid_drce, cost_grid_drce,
                                   alpha=0.5, color='green')
    
    # Add a legend.
    surfaces = [surface_lqg, surface_wdrc, surface_drce]
    labels = ['LQG (infinite)', 'WDRC (infinite)', 'DRCE (infinte)']
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
    
    # Save the figure in the same base directory.
    save_path = os.path.join(base_path, 'experiment_{}_{}_comparison.pdf'.format(dist, noise_dist))
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str,
                        help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', required=False, default="normal", type=str,
                        help="Noise distribution (normal or quadratic)")
    parser.add_argument('--use_lambda', required=False, action="store_true",
                        help="Use lambda-based experiment")
    args = parser.parse_args()
    
    # For this experiment we assume lambda mode; call the corresponding plotting function.
    plot_experiment_lambda3(args.dist, args.noise_dist)

if __name__ == "__main__":
    main()
