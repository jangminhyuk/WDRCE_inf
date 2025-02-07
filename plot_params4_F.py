#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file generates Figure 3(a), (b)

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import re
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d

def summarize_lambda(wdrc_drkf_lambda_values, wdrc_drkf_theta_v_values, wdrc_drkf_cost_values ,wdrc_lambda_values, wdrc_theta_v_values, wdrc_cost_values , drce_lambda_values, drce_theta_v_values, drce_cost_values, drcmmse_lambda_values, drcmmse_theta_v_values, drcmmse_cost_values,  dist, noise_dist,use_lambda, path):
    
    surfaces = []
    labels = []
    # Create 3D plot
    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    })
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    
    # Repeat the process for WDRC
    # Interpolate cost values for smooth surface - WDRC
    lambda_grid_wdrc, theta_v_grid_wdrc = np.meshgrid(
    np.linspace(min(wdrc_lambda_values), max(wdrc_lambda_values), 100),
    np.linspace(min(wdrc_theta_v_values), max(wdrc_theta_v_values), 100)
    )
    cost_grid_wdrc = griddata(
        (wdrc_lambda_values, wdrc_theta_v_values), wdrc_cost_values,
        (lambda_grid_wdrc, theta_v_grid_wdrc), method='linear'  # Use linear interpolation
    )

    # Plot smooth surface - WDRC
    surface_wdrc =ax.plot_surface(lambda_grid_wdrc, theta_v_grid_wdrc, cost_grid_wdrc, alpha=0.5, color='blue', label='WDRC')
    surfaces.append(surface_wdrc)
    labels.append('WDRC [12]')
    #--------------
    # Interpolate cost values for smooth surface - DRCMMSE
    lambda_grid_drcmmse, theta_v_grid_drcmmse = np.meshgrid(
        np.linspace(min(drcmmse_lambda_values), max(drcmmse_lambda_values), 100),
        np.linspace(min(drcmmse_theta_v_values), max(drcmmse_theta_v_values), 100)
    )
    cost_grid_drcmmse = griddata(
        (drcmmse_lambda_values, drcmmse_theta_v_values), drcmmse_cost_values,
        (lambda_grid_drcmmse, theta_v_grid_drcmmse), method='cubic'
    )
    
    # Plot smooth surface - DCE
    surface_drcmmse = ax.plot_surface(lambda_grid_drcmmse, theta_v_grid_drcmmse, cost_grid_drcmmse, alpha=0.6, color='yellow', label='WDRC+DRMMSE', antialiased=False)
    surfaces.append(surface_drcmmse)
    labels.append('WDRC + DRMMSE [31]')
    #--------------
    # Interpolate cost values for smooth surface - WDRC DRKF
    lambda_grid_wdrc_drkf, theta_v_grid_wdrc_drkf = np.meshgrid(
        np.linspace(min(wdrc_drkf_lambda_values), max(wdrc_drkf_lambda_values), 100),
        np.linspace(min(wdrc_drkf_theta_v_values), max(wdrc_drkf_theta_v_values), 100)
    )
    cost_grid_wdrc_drkf = griddata(
        (wdrc_drkf_lambda_values, wdrc_drkf_theta_v_values), wdrc_drkf_cost_values,
        (lambda_grid_wdrc_drkf, theta_v_grid_wdrc_drkf), method='cubic'
    )
    
    # Plot smooth surface - DCE
    surface_wdrc_drkf = ax.plot_surface(lambda_grid_wdrc_drkf, theta_v_grid_wdrc_drkf, cost_grid_wdrc_drkf, alpha=0.4, color='red', label='WDRC+DRKF')
    surfaces.append(surface_wdrc_drkf)
    labels.append('WDRC + WKF [28]')
    
    
    #---------------------------
    # Interpolate cost values for smooth surface - DRCE
    lambda_grid_drce, theta_v_grid_drce = np.meshgrid(
        np.linspace(min(drce_lambda_values), max(drce_lambda_values), 100),
        np.linspace(min(drce_theta_v_values), max(drce_theta_v_values), 100)
    )
    cost_grid_drce = griddata(
        (drce_lambda_values, drce_theta_v_values), drce_cost_values,
        (lambda_grid_drce, theta_v_grid_drce), method='cubic'
    )
    
    # Plot smooth surface - DCE
    surface_drce = ax.plot_surface(lambda_grid_drce, theta_v_grid_drce, cost_grid_drce, alpha=0.6, color='green', label='WDR-CE')
    surfaces.append(surface_drce)
    labels.append('WDR-CE [Ours]')
    
    
    legend = fig.legend(
        handles=surfaces,
        labels=labels,
        bbox_to_anchor=(0.67, 0.86),  # Moves legend further out of the plot
        loc='center right',
        frameon=True,
        framealpha=1.0,
        facecolor='white',
        fontsize=18,  # Reduced fontsize to make legend more compact
        borderpad=0.3,
        handletextpad=0.2,  # Reduce space between legend handle and text
        labelspacing=0.1    # Reduce vertical space between entries
    )
    legend.get_frame().set_alpha(0.7)
    legend.get_frame().set_facecolor('white')
    
    # Set labels
    ax.set_xlabel(r'$\lambda$', fontsize=24, labelpad=8)
    ax.set_ylabel(r'$\theta$', fontsize=24, labelpad=8)
    ax.set_zlabel(r'Total Cost', fontsize=24, rotation=90, labelpad=8)
    ax.set_yticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10])
    ax.set_ylim([0.5, 10.5])
    
    z_min = np.min([cost_grid_drce, cost_grid_wdrc, cost_grid_wdrc_drkf, cost_grid_drcmmse])
    z_max = np.max([cost_grid_drce, cost_grid_wdrc, cost_grid_wdrc_drkf, cost_grid_drcmmse])
    
    if dist=="quadratic":
        z_ticks = np.arange(int(np.floor(z_min)), int(np.ceil(z_max))+100, step=100)
    else:
        z_ticks = np.linspace(int(np.floor(z_min)), int(np.ceil(z_max)), num=5)
    
    z_ticks = [int(tick) for tick in z_ticks]

    # Set the z-ticks on the plot
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f'${int(tick)}$' for tick in z_ticks], ha='center', va='center')

    ax.tick_params(axis='z', which='major', labelsize=18, pad=4)  # Add padding between z ticks and axis
    ax.tick_params(axis='x', which='major', labelsize=18, pad=0)  # Add padding between z ticks and axis
    ax.tick_params(axis='y', which='major', labelsize=18, pad=0)  # Add padding between z ticks and axis
    
    
    
    ax.zaxis.set_rotate_label(False)
    a = ax.zaxis.label.get_rotation()
    if a<180:
        a += 00
    ax.zaxis.label.set_rotation(a)
    a = ax.zaxis.label.get_rotation()
    ax.view_init(elev=15, azim=40)
    #ax.xaxis._axinfo['label']['space_factor'] = 2.8
    
    plt.show()
    fig.savefig(path + 'params_{}_{}_9.pdf'.format(dist, noise_dist), dpi=300, bbox_inches="tight", pad_inches=0.3)
    #plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    parser.add_argument('--use_lambda', required=False, default="True", action="store_true") #use lambda results if flagged
    args = parser.parse_args()
    
    
    if args.use_lambda:
        path = "./results/{}_{}/finite/multiple/params_lambda/filter/".format(args.dist, args.noise_dist)
    else:
        path = "./results/{}_{}/finite/multiple/params_thetas/filter/".format(args.dist, args.noise_dist)

    #Load data
    drcmmse_theta_w_values =[]
    drcmmse_lambda_values = []
    drcmmse_theta_v_values = []
    drcmmse_cost_values = []
    
    drce_theta_w_values =[]
    drce_lambda_values = []
    drce_theta_v_values = []
    drce_cost_values = []
    
    wdrc_theta_w_values = []
    wdrc_lambda_values = []
    wdrc_theta_v_values = []
    wdrc_cost_values = []
    
    wdrc_drkf_theta_w_values =[]
    wdrc_drkf_lambda_values = []
    wdrc_drkf_theta_v_values = []
    wdrc_drkf_cost_values = []
    # theta_v_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    # lambda_list = [ 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    
    # TODO : Modify the theta_v_list and lambda_list below to match your experiments!!! 

    #theta_v_list = [2.0, 4.0, 6.0, 8.0, 10.0]
    theta_v_list = [2.0, 4.0, 6.0, 8.0, 10.0]
    theta_w_list = [2.0, 4.0, 6.0, 8.0, 10.0]
    if args.dist=='normal':
        lambda_list = [10, 20, 30, 40, 50] # disturbance distribution penalty parameter
    else:
        lambda_list = [10, 20, 30, 40, 50] # disturbance distribution penalty parameter

    # Regular expression pattern to extract numbers from file names
    
    if args.use_lambda:
        pattern_drce = r"drce_(\d+)and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_drcmmse = r"drcmmse_(\d+)and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_wdrc_drkf = r"wdrc_drkf(\d+)and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_wdrc = r"wdrc_(\d+)"
    else:
        pattern_drcmmse = r"drcmmse_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_drce = r"drce_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_wdrc_drkf = r"wdrc_drkf_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_wdrc = r"wdrc_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
    #pattern_lqg = r"lqg.pkl"
    # Iterate over each file in the directory
    for filename in os.listdir(path):
        match = re.search(pattern_drce, filename)
        if match:
            if args.use_lambda:
                lambda_value = float(match.group(1))  # Extract lambda
                theta_v_value = float(match.group(2))   # Extract theta_v value
                theta_v_str = match.group(3)
                theta_v_value += float(theta_v_str)/10
                #changed _1_5_ to 1.5!
                # Store theta_w and theta values
                drce_lambda_values.append(lambda_value)
                drce_theta_v_values.append(theta_v_value)
            else:
                theta_w_value = float(match.group(1))  # Extract theta_w value
                theta_w_str = match.group(2)
                theta_w_value += float(theta_w_str)/10
                theta_v_value = float(match.group(3))   # Extract theta_v value
                theta_v_str = match.group(4)
                theta_v_value += float(theta_v_str)/10
                #changed _1_5_ to 1.5!
                # Store theta_w and theta values
                drce_theta_w_values.append(theta_w_value)
                drce_theta_v_values.append(theta_v_value)
            
            drce_file = open(path + filename, 'rb')
            drce_cost = pickle.load(drce_file)
            drce_file.close()
            drce_cost_values.append(drce_cost[0])  # Store cost value
        else:
            match_drcmmse = re.search(pattern_drcmmse, filename)
            if match_drcmmse:
                if args.use_lambda:
                    lambda_value = float(match_drcmmse.group(1))  # Extract lambda
                    theta_v_value = float(match_drcmmse.group(2))   # Extract theta_v value
                    theta_v_str = match_drcmmse.group(3)
                    theta_v_value += float(theta_v_str)/10
                    #changed _1_5_ to 1.5!
                    # Store theta_w and theta values
                    drcmmse_lambda_values.append(lambda_value)
                    drcmmse_theta_v_values.append(theta_v_value)
                else:
                    theta_w_value = float(match_drcmmse.group(1))  # Extract theta_w value
                    theta_w_str = match_drcmmse.group(2)
                    theta_w_value += float(theta_w_str)/10
                    theta_v_value = float(match_drcmmse.group(3))   # Extract theta_v value
                    theta_v_str = match_drcmmse.group(4)
                    theta_v_value += float(theta_v_str)/10
                    #changed _1_5_ to 1.5!
                    # Store theta_w and theta values
                    drcmmse_theta_w_values.append(theta_w_value)
                    drcmmse_theta_v_values.append(theta_v_value)
                
                drcmmse_file = open(path + filename, 'rb')
                drcmmse_cost = pickle.load(drcmmse_file)
                drcmmse_file.close()
                drcmmse_cost_values.append(drcmmse_cost[0])  # Store cost value
            else:
                match_wdrc_drkf = re.search(pattern_wdrc_drkf, filename)
                if match_wdrc_drkf:
                    if args.use_lambda:
                        lambda_value = float(match_wdrc_drkf.group(1))  # Extract lambda
                        theta_v_value = float(match_wdrc_drkf.group(2))   # Extract theta_v value
                        theta_v_str = match_wdrc_drkf.group(3)
                        theta_v_value += float(theta_v_str)/10
                        #changed _1_5_ to 1.5!
                        # Store theta_w and theta values
                        wdrc_drkf_lambda_values.append(lambda_value)
                        wdrc_drkf_theta_v_values.append(theta_v_value)
                        
                    else:
                        theta_w_value = float(match_wdrc_drkf.group(1))  # Extract theta_w value
                        theta_w_str = match_wdrc_drkf.group(2)
                        theta_w_value += float(theta_w_str)/10
                        theta_v_value = float(match_wdrc_drkf.group(3))   # Extract theta_v value
                        theta_v_str = match_wdrc_drkf.group(4)
                        theta_v_value += float(theta_v_str)/10
                        #changed _1_5_ to 1.5!
                        # Store theta_w and theta values
                        
                        wdrc_drkf_theta_w_values.append(theta_w_value)
                        wdrc_drkf_theta_v_values.append(theta_v_value)
                    
                    wdrc_drkf_file = open(path + filename, 'rb')
                    wdrc_drkf_cost = pickle.load(wdrc_drkf_file)
                    wdrc_drkf_file.close()
                    wdrc_drkf_cost_values.append(wdrc_drkf_cost[0])  # Store cost value
                else:
                    match_wdrc = re.search(pattern_wdrc, filename)
                    if match_wdrc: # wdrc
                        if args.use_lambda:
                            lambda_value = float(match_wdrc.group(1))  # Extract lambda
                            
                        else:
                            theta_w_value = float(match_wdrc.group(1))  # Extract theta_w value
                            theta_w_str = match_wdrc.group(2)
                            theta_w_value += float(theta_w_str)/10
                        
                        wdrc_file = open(path + filename, 'rb')
                        wdrc_cost = pickle.load(wdrc_file)
                        wdrc_file.close()
                        
                        for aux_theta_v in theta_v_list:
                            if args.use_lambda:
                                wdrc_lambda_values.append(lambda_value)
                            else:
                                wdrc_theta_w_values.append(theta_w_value)
                            
                            wdrc_theta_v_values.append(aux_theta_v) # since wdrc not affected by theta v, just add auxilary theta for plot
                            wdrc_cost_values.append(wdrc_cost[0])
                            
                
                
                    

    # Convert lists to numpy arrays
    if args.use_lambda:
        drcmmse_lambda_values = np.array(drcmmse_lambda_values)
        drce_lambda_values = np.array(drce_lambda_values)
        wdrc_lambda_values = np.array(wdrc_lambda_values)
        wdrc_drkf_lambda_values = np.array(wdrc_drkf_lambda_values)
    else:
        drcmmse_theta_w_values = np.array(drcmmse_theta_w_values)
        drce_theta_w_values = np.array(drce_theta_w_values)
        wdrc_theta_w_values = np.array(wdrc_theta_w_values)
        wdrc_drkf_theta_w_values = np.array(wdrc_drkf_theta_w_values)
    
    drcmmse_theta_v_values = np.array(drcmmse_theta_v_values)
    drcmmse_cost_values = np.array(drcmmse_cost_values)
    
    drce_theta_v_values = np.array(drce_theta_v_values)
    drce_cost_values = np.array(drce_cost_values)

    wdrc_theta_v_values = np.array(wdrc_theta_v_values)
    wdrc_cost_values = np.array(wdrc_cost_values)
    
    wdrc_drkf_theta_v_values = np.array(wdrc_drkf_theta_v_values)
    wdrc_drkf_cost_values = np.array(wdrc_drkf_cost_values)
    
    #print("wdrcdrkf ", wdrc_drkf_theta_w_values)
    
    summarize_lambda(wdrc_drkf_lambda_values, wdrc_drkf_theta_v_values, wdrc_drkf_cost_values ,wdrc_lambda_values, wdrc_theta_v_values, wdrc_cost_values , drce_lambda_values, drce_theta_v_values, drce_cost_values, drcmmse_lambda_values, drcmmse_theta_v_values, drcmmse_cost_values, args.dist, args.noise_dist,  args.use_lambda, path)
    
