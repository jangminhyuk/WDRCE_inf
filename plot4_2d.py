#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import re
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
from matplotlib import cm
from scipy.interpolate import interp1d

def summarize(out_lq_list,  out_wdrc_list, out_drce_list, dist, noise_dist, path, num, trajectory, plot_results=True):
    x_lqr_list,  J_lqr_list, y_lqr_list, u_lqr_list, traj_lqr_list  = [], [], [], [], []
    x_wdrc_list, J_wdrc_list, y_wdrc_list, u_wdrc_list = [], [], [], [] # original wdrc with ordinary Kalman Filter
    x_drce_list, J_drce_list, y_drce_list, u_drce_list, traj_drce_list = [], [], [], [], [] # drce
    time_wdrc_list, time_lqr_list, time_drce_list, time_drlqc_list = [], [], [], []


    for out in out_lq_list:
         x_lqr_list.append(out['state_traj'])
         J_lqr_list.append(out['cost'])
         y_lqr_list.append(out['output_traj'])
         u_lqr_list.append(out['control_traj'])
         time_lqr_list.append(out['comp_time'])
         traj_lqr_list.append(out['desired_traj'])
         
         
    x_lqr_mean, J_lqr_mean, y_lqr_mean, u_lqr_mean = np.mean(x_lqr_list, axis=0), np.mean(J_lqr_list, axis=0), np.mean(y_lqr_list, axis=0), np.mean(u_lqr_list, axis=0)
    x_lqr_std, J_lqr_std, y_lqr_std, u_lqr_std = np.std(x_lqr_list, axis=0), np.std(J_lqr_list, axis=0), np.std(y_lqr_list, axis=0), np.std(u_lqr_list, axis=0)
    time_lqr_ar = np.array(time_lqr_list)
    print("LQG cost : ", J_lqr_mean[0])
    print("LQG cost std : ", J_lqr_std[0])
    J_lqr_ar = np.array(J_lqr_list)
    
    
    for out in out_wdrc_list:
        x_wdrc_list.append(out['state_traj'])
        J_wdrc_list.append(out['cost'])
        y_wdrc_list.append(out['output_traj'])
        u_wdrc_list.append(out['control_traj'])
        time_wdrc_list.append(out['comp_time'])
    x_wdrc_mean, J_wdrc_mean, y_wdrc_mean, u_wdrc_mean = np.mean(x_wdrc_list, axis=0), np.mean(J_wdrc_list, axis=0), np.mean(y_wdrc_list, axis=0), np.mean(u_wdrc_list, axis=0)
    x_wdrc_std, J_wdrc_std, y_wdrc_std, u_wdrc_std = np.std(x_wdrc_list, axis=0), np.std(J_wdrc_list, axis=0), np.std(y_wdrc_list, axis=0), np.std(u_wdrc_list, axis=0)
    time_wdrc_ar = np.array(time_wdrc_list)
    print("WDRC cost : ", J_wdrc_mean[0])
    print("WDRC cost std : ", J_wdrc_std[0])
    J_wdrc_ar = np.array(J_wdrc_list)



    for out in out_drce_list:
        x_drce_list.append(out['state_traj'])
        J_drce_list.append(out['cost'])
        y_drce_list.append(out['output_traj'])
        u_drce_list.append(out['control_traj'])
        time_drce_list.append(out['comp_time'])
        traj_drce_list.append(out['desired_traj'])
          
        
    x_drce_mean, J_drce_mean, y_drce_mean, u_drce_mean = np.mean(x_drce_list, axis=0), np.mean(J_drce_list, axis=0), np.mean(y_drce_list, axis=0), np.mean(u_drce_list, axis=0)
    x_drce_std, J_drce_std, y_drce_std, u_drce_std = np.std(x_drce_list, axis=0), np.std(J_drce_list, axis=0), np.std(y_drce_list, axis=0), np.std(u_drce_list, axis=0)
    time_drce_ar = np.array(time_drce_list)
    print("DRCE cost : ", J_drce_mean[0])
    print("DRCE cost std : ", J_drce_std[0])
    J_drce_ar = np.array(J_drce_list)   
    nx = x_drce_mean.shape[1]
    T = u_drce_mean.shape[0]
    
    costs = [sum( 5*(traj_lqr_list[idx][2, :] - x_drce_list[idx][:-1, 2, 0])**2 + (traj_lqr_list[idx][0, :] - x_drce_list[idx][:-1, 0, 0])**2) for idx in range(len(x_lqr_list))]
    idx = np.argmin(costs)
    idx = 0
    # 2D plot
    # Plotting the results with improved visualization
    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    })
    fig = plt.figure(figsize=(10, 6))
    # Plot the actual tracked trajectory # First element in the list!
    y_max = np.max(np.concatenate([x_lqr_list[idx][:, 0, 0],  traj_lqr_list[idx][0,:],  x_wdrc_list[idx][:, 0, 0],  x_drce_list[idx][:, 0, 0]]))
    y_min = np.min(np.concatenate([x_lqr_list[idx][:, 0, 0],  traj_lqr_list[idx][0,:],  x_wdrc_list[idx][:, 0, 0],  x_drce_list[idx][:, 0, 0]]))

    plt.plot(x_lqr_list[idx][:, 2, 0], x_lqr_list[idx][:, 0, 0], label='LQG', color='red', linewidth=2)
    plt.plot(x_wdrc_list[idx][:, 2, 0], x_wdrc_list[idx][:, 0, 0], label='WDRC [12]', color='blue', linewidth=2)
    plt.plot(x_drce_list[idx][:, 2, 0], x_drce_list[idx][:, 0, 0], label='WDR-CE [Ours]', color='green', linewidth=2)
    
    # Plot the desired trajectory based on the selected type
    plt.plot(traj_lqr_list[idx][2, :], traj_lqr_list[idx][0, :], label='Desired Trajectory', color='black', linewidth=2)

    # Highlight the start and end points
    plt.scatter(traj_lqr_list[0][2, 0], traj_lqr_list[0][0, 0], color='black', marker='X', s=100)#, label='Start/End Position')
    plt.scatter(traj_lqr_list[0][2, -1], traj_lqr_list[0][0, -1], color='black', marker='X', s=100)

    plt.ylim([1.1*y_min, 1.1*y_max])
    # Label the axes
    plt.xlabel('X Position [m]', fontsize=28)
    plt.ylabel('Y Position [m]', fontsize=28)
    plt.xticks(fontsize=22) 
    plt.yticks(fontsize=22)
    
    # Set the aspect ratio to be equal so the plot looks correct
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Add a grid for better visibility
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize the legend position and style
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.27),  # Fine-tune position to be above and centered
        frameon=True,
        framealpha=1.0,
        facecolor='white',
        fontsize=22,  # Reduced fontsize to make legend more compact
        borderpad=0.3,
        handletextpad=0.3,  # Reduce space between legend handle and text
        labelspacing=0.15,   # Reduce vertical space between entries
        ncol=2  # Make the legend horizontal by setting number of columns
    )
    legend.get_frame().set_alpha(0.7)
    legend.get_frame().set_facecolor('white')

    # Save the plot before displaying
    plt.tight_layout()
    plt.savefig(path + '2D_{}_{}_{}.pdf'.format(dist, noise_dist, trajectory), dpi=300, bbox_inches="tight")

    # Display the plot
    plt.show()

    # Clear the figure
    plt.clf()
    
    
    # ------------------------------------------------------------
    # if plot_results:
    #     nx = x_drce_mean.shape[1]
    #     T = u_drce_mean.shape[0]
    #     nu = u_drce_mean.shape[1]
    #     ny= y_drce_mean.shape[1]

    #     fig = plt.figure(figsize=(6,4), dpi=300)

    #     t = np.arange(T+1)
    #     for i in range(nx):

    #         if x_lqr_list != []:
    #             plt.plot(t, x_lqr_mean[:,i,0], 'tab:red', label='LQG')
    #             plt.fill_between(t, x_lqr_mean[:,i, 0] + 0.3*x_lqr_std[:,i,0],
    #                            x_lqr_mean[:,i,0] - 0.3*x_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
    #         if x_wdrc_list != []:
    #             plt.plot(t, x_wdrc_mean[:,i,0], 'tab:blue', label='WDRC')
    #             plt.fill_between(t, x_wdrc_mean[:,i,0] + 0.3*x_wdrc_std[:,i,0],
    #                             x_wdrc_mean[:,i,0] - 0.3*x_wdrc_std[:,i,0], facecolor='tab:blue', alpha=0.3)
    #         if x_drce_list != []:
    #             plt.plot(t, x_drce_mean[:,i,0], 'tab:green', label='WDR-CE')
    #             plt.fill_between(t, x_drce_mean[:,i, 0] + 0.3*x_drce_std[:,i,0],
    #                            x_drce_mean[:,i,0] - 0.3*x_drce_std[:,i,0], facecolor='tab:green', alpha=0.3)
            
                
    #         plt.xlabel(r'$t$', fontsize=22)
    #         plt.ylabel(r'$x_{{{}}}$'.format(i+1), fontsize=22)
    #         plt.legend(fontsize=20)
    #         plt.grid()
    #         plt.xticks(fontsize=20)
    #         plt.yticks(fontsize=20)
    #         plt.xlim([t[0], t[-1]])
    #         ax = fig.gca()
    #         ax.locator_params(axis='y', nbins=5)
    #         ax.locator_params(axis='x', nbins=5)
    #         fig.set_size_inches(6, 4)
    #         plt.savefig(path +'states_{}_{}_{}_{}_{}.pdf'.format(i+1, num, dist, noise_dist, trajectory), dpi=300, bbox_inches="tight")
    #         plt.clf()

    #     t = np.arange(T)
    #     for i in range(nu):

    #         if u_lqr_list != []:
    #             plt.plot(t, u_lqr_mean[:,i,0], 'tab:red', label='LQG')
    #             plt.fill_between(t, u_lqr_mean[:,i,0] + 0.25*u_lqr_std[:,i,0],
    #                          u_lqr_mean[:,i,0] - 0.25*u_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
    #         if u_wdrc_list != []:
    #             plt.plot(t, u_wdrc_mean[:,i,0], 'tab:blue', label='WDRC')
    #             plt.fill_between(t, u_wdrc_mean[:,i,0] + 0.25*u_wdrc_std[:,i,0],
    #                             u_wdrc_mean[:,i,0] - 0.25*u_wdrc_std[:,i,0], facecolor='tab:blue', alpha=0.3)
    #         if u_drce_list != []:
    #             plt.plot(t, u_drce_mean[:,i,0], 'tab:green', label='WDR-CE')
    #             plt.fill_between(t, u_drce_mean[:,i,0] + 0.25*u_drce_std[:,i,0],
    #                          u_drce_mean[:,i,0] - 0.25*u_drce_std[:,i,0], facecolor='tab:green', alpha=0.3)       
            
    #         plt.xlabel(r'$t$', fontsize=16)
    #         plt.ylabel(r'$u_{{{}}}$'.format(i+1), fontsize=16)
    #         plt.legend(fontsize=16)
    #         plt.grid()
    #         plt.xticks(fontsize=20)
    #         plt.yticks(fontsize=20)
    #         plt.xlim([t[0], t[-1]])
    #         ax = fig.gca()
    #         ax.locator_params(axis='y', nbins=5)
    #         ax.locator_params(axis='x', nbins=5)
    #         fig.set_size_inches(6, 4)
    #         plt.savefig(path +'controls_{}_{}_{}_{}_{}.pdf'.format(i+1, num, dist, noise_dist, trajectory), dpi=300, bbox_inches="tight")
    #         plt.clf()

    #     t = np.arange(T+1)
    #     for i in range(ny):
    #         if y_lqr_list != []:
    #             plt.plot(t, y_lqr_mean[:,i,0], 'tab:red', label='LQG')
    #             plt.fill_between(t, y_lqr_mean[:,i,0] + 0.25*y_lqr_std[:,i,0],
    #                          y_lqr_mean[:,i, 0] - 0.25*y_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
    #         if y_wdrc_list != []:
    #             plt.plot(t, y_wdrc_mean[:,i,0], 'tab:blue', label='WDRC')
    #             plt.fill_between(t, y_wdrc_mean[:,i,0] + 0.25*y_wdrc_std[:,i,0],
    #                             y_wdrc_mean[:,i, 0] - 0.25*y_wdrc_std[:,i,0], facecolor='tab:blue', alpha=0.3)
    #         if y_drce_list != []:
    #             plt.plot(t, y_drce_mean[:,i,0], 'tab:green', label='WDR-CE')
    #             plt.fill_between(t, y_drce_mean[:,i,0] + 0.25*y_drce_std[:,i,0],
    #                          y_drce_mean[:,i, 0] - 0.25*y_drce_std[:,i,0], facecolor='tab:green', alpha=0.3)
            
    #         plt.xlabel(r'$t$', fontsize=16)
    #         plt.ylabel(r'$y_{{{}}}$'.format(i+1), fontsize=16)
    #         plt.legend(fontsize=16)
    #         plt.grid()
    #         plt.xticks(fontsize=20)
    #         plt.yticks(fontsize=20)
    #         plt.xlim([t[0], t[-1]])
    #         ax = fig.gca()
    #         ax.locator_params(axis='y', nbins=5)
    #         ax.locator_params(axis='x', nbins=5)
    #         fig.set_size_inches(6, 4)
    #         plt.savefig(path +'outputs_{}_{}_{}_{}_{}.pdf'.format(i+1,num, dist, noise_dist, trajectory), dpi=300, bbox_inches="tight")
    #         plt.clf()


    #     plt.title('Optimal Value')
    #     t = np.arange(T+1)

    #     if J_lqr_list != []:
    #         plt.plot(t, J_lqr_mean, 'tab:red', label='LQG')
    #         plt.fill_between(t, J_lqr_mean + 0.25*J_lqr_std, J_lqr_mean - 0.25*J_lqr_std, facecolor='tab:red', alpha=0.3)
    #     if J_wdrc_list != []:
    #         plt.plot(t, J_wdrc_mean, 'tab:blue', label='WDRC')
    #         plt.fill_between(t, J_wdrc_mean + 0.25*J_wdrc_std, J_wdrc_mean - 0.25*J_wdrc_std, facecolor='tab:blue', alpha=0.3)
    #     if J_drce_list != []:
    #         plt.plot(t, J_drce_mean, 'tab:green', label='WDR-CE')
    #         plt.fill_between(t, J_drce_mean + 0.25*J_drce_std, J_drce_mean - 0.25*J_drce_std, facecolor='tab:green', alpha=0.3)
        
    #     plt.xlabel(r'$t$', fontsize=16)
    #     plt.ylabel(r'$V_t(x_t)$', fontsize=16)
    #     plt.legend(fontsize=16)
    #     plt.grid()
    #     plt.xlim([t[0], t[-1]])
    #     plt.xticks(fontsize=16)
    #     plt.yticks(fontsize=16)
    #     plt.savefig(path +'J_{}_{}_{}_{}.pdf'.format(num, dist, noise_dist, trajectory), dpi=300, bbox_inches="tight")
    #     plt.clf()


    #     # Plot the histograms
    #     ax = fig.gca()
    #     t = np.arange(T+1)

    #     max_bin = np.max([J_lqr_ar[:,0], J_wdrc_ar[:,0], J_drce_ar[:,0]])
    #     min_bin = np.min([J_lqr_ar[:,0], J_wdrc_ar[:,0], J_drce_ar[:,0]])

    #     # Plot histograms for LQG and WDR-CE
    #     ax.hist(J_lqr_ar[:,0], bins=50, range=(min_bin, max_bin), color='tab:red', label='LQG', alpha=0.5, linewidth=0.5, edgecolor='tab:red')
    #     ax.hist(J_wdrc_ar[:,0], bins=50, range=(min_bin, max_bin), color='tab:blue', label='WDRC [12]', alpha=0.5, linewidth=0.5, edgecolor='tab:blue')
    #     ax.hist(J_drce_ar[:,0], bins=50, range=(min_bin, max_bin), color='tab:green', label='WDR-CE', alpha=0.5, linewidth=0.5, edgecolor='tab:green')

    #     # Add vertical lines for means
    #     ax.axvline(J_lqr_ar[:,0].mean(), color='maroon', linestyle='dashed', linewidth=1.5)
    #     ax.axvline(J_wdrc_ar[:,0].mean(), color='blue', linestyle='dashed', linewidth=1.5)
    #     ax.axvline(J_drce_ar[:,0].mean(), color='green', linestyle='dashed', linewidth=1.5)

    #     # Set x-axis to scientific notation
    #     ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1000:.0f}'))

    #     # Increase fontsize for tick labels
    #     plt.xticks(fontsize=16)
    #     plt.yticks(fontsize=16)

    #     # Legend settings
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     order = [0, 1]
    #     ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)

    #     # Grid and labels
    #     ax.grid()
    #     ax.set_axisbelow(True)
    #     plt.xlabel(r'Total Cost (x1000)', fontsize=16)
    #     plt.ylabel(r'Frequency', fontsize=16)

    #     # Save the plot
    #     plt.savefig(path + 'J_hist_{}_{}_{}_{}.pdf'.format(num, dist, noise_dist, trajectory), dpi=300, bbox_inches="tight")
    #     plt.clf()
    #     plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    parser.add_argument('--num_sim', required=False, default=500, type=int)
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    parser.add_argument('--use_lambda', required=False, action="store_true") #use lambda results if flagged
    parser.add_argument('--trajectory', required=False, default="curvy", type=str)
    args = parser.parse_args()

    horizon = "finite"
        
    if args.use_lambda:
        path = "./results/{}_{}/finite/multiple/params_lambda/2d_{}/".format(args.dist, args.noise_dist, args.trajectory)
        rawpath = "./results/{}_{}/finite/multiple/params_lambda/2d_{}/raw/".format(args.dist, args.noise_dist, args.trajectory)
    else:
        path = "./results/{}_{}/finite/multiple/params_thetas/2d_{}/".format(args.dist, args.noise_dist, args.trajectory)
        rawpath = "./results/{}_{}/finite/multiple/params_thetas/2d_{}/raw/".format(args.dist, args.noise_dist, args.trajectory)

    #Load data
    drlqc_theta_w_values =[]
    drlqc_lambda_values = []
    drlqc_theta_v_values = []
    drlqc_cost_values = []
    
    drce_theta_w_values =[]
    drce_lambda_values = []
    drce_theta_v_values = []
    drce_cost_values = []
    
    wdrc_theta_w_values = []
    wdrc_lambda_values = []
    wdrc_theta_v_values = []
    wdrc_cost_values = []
    
    lqg_theta_w_values =[]
    lqg_lambda_values = []
    lqg_theta_v_values = []
    lqg_cost_values = []
    
    drlqc_optimal_theta_w, drlqc_optimal_theta_v, drlqc_optimal_cost = 0, 0, np.inf
    drce_optimal_theta_w, drce_optimal_theta_v, drce_optimal_cost = 0, 0, np.inf
    wdrc_optimal_theta_w, wdrc_optimal_cost = 0, np.inf
    drce_optimal_lambda, wdrc_optimal_lambda = 0, 0
    
    if args.dist=='normal':
        lambda_list = [12, 15, 20, 25, 30, 35, 40, 45, 50] # disturbance distribution penalty parameter
        theta_v_list = [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        theta_w_list = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    else:
        lambda_list = [15, 20, 25, 30, 35, 40, 45, 50] # disturbance distribution penalty parameter
        theta_v_list = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
        theta_w_list = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
        
    
    # Regular expression pattern to extract numbers from file names
    if args.use_lambda:
        pattern_drce = r"drce_(\d+)and_(\d+_\d+)_?(\d+_\d+)?"
        pattern_drlqc = r"drlqc_(\d+_\d+)_?(\d+_\d+)?and_(\d+_\d+)_?(\d+_\d+)?"
        pattern_wdrc = r"wdrc_(\d+)"
    else:
        pattern_drlqc = r"drlqc_(\d+_\d+)_?(\d+_\d+)?and_(\d+_\d+)_?(\d+_\d+)?"
        pattern_drce = r"drce_(\d+_\d+)_?(\d+_\d+)?and_(\d+_\d+)_?(\d+_\d+)?"
        pattern_wdrc = r"wdrc_(\d+_\d+)_?(\d+_\d+)?"
    pattern_lqg = r"lqg.pkl"

    # Function to convert underscore-separated values (e.g., '0_01') to float
    def convert_to_float(underscore_value):
        return float(underscore_value.replace('_', '.'))

    # Iterate over each file in the directory
    for filename in os.listdir(path):
        match = re.search(pattern_drce, filename)
        if match:
            if args.use_lambda:
                lambda_value = (match.group(1))  # Extract lambda and convert to float
                theta_v_value = convert_to_float(match.group(2))  # Extract theta_v value and convert to float
                # Store lambda and theta_v values
                drce_lambda_values.append(lambda_value)
                drce_theta_v_values.append(theta_v_value)
            else:
                theta_w_value = convert_to_float(match.group(1))  # Extract theta_w value and convert to float
                if match.group(2):
                    theta_w_value += convert_to_float(match.group(2))
                theta_v_value = convert_to_float(match.group(3))  # Extract theta_v value and convert to float
                if match.group(4):
                    theta_v_value += convert_to_float(match.group(4))
                # Store theta_w and theta_v values
                drce_theta_w_values.append(theta_w_value)
                drce_theta_v_values.append(theta_v_value)
            
            drce_file = open(path + filename, 'rb')
            drce_cost = pickle.load(drce_file)
            if drce_cost[0] < drce_optimal_cost:
                drce_optimal_cost = drce_cost[0]
                if args.use_lambda:
                    drce_optimal_lambda = lambda_value
                else:
                    drce_optimal_theta_w = theta_w_value
                drce_optimal_theta_v = theta_v_value
            drce_file.close()
            drce_cost_values.append(drce_cost[0])  # Store cost value
        
        else:
            match_lqg = re.search(pattern_lqg, filename)
            if match_lqg:
                lqg_file = open(path + filename, 'rb')
                lqg_cost = pickle.load(lqg_file)
                lqg_file.close()
                if args.use_lambda:
                    for aux_lambda in lambda_list:
                        for aux_theta_v in theta_v_list:
                            lqg_lambda_values.append(aux_lambda)
                            lqg_theta_v_values.append(aux_theta_v)
                            lqg_cost_values.append(lqg_cost[0])
                else:
                    for aux_theta_w in theta_w_list:
                        for aux_theta_v in theta_v_list:
                            lqg_theta_w_values.append(aux_theta_w)
                            lqg_theta_v_values.append(aux_theta_v)
                            lqg_cost_values.append(lqg_cost[0])

                
    # choose the parameter to draw!
    if args.trajectory =='curvy':
        drce_theta_v = 5.0 #5.0
        drce_lambda = 30000 #30000
    elif args.trajectory =='circular':
        drce_theta_v = 4.0
        drce_lambda = 20000
        
    drce_theta_v_str = str(drce_theta_v).replace('.', '_')
    
    if args.use_lambda:
        drce_filename = f"drce_{str(drce_lambda)}and_{drce_theta_v_str}.pkl"
        drce_filepath = rawpath + drce_filename
    drce_filepath = rawpath + drce_filename
    
    if args.use_lambda:
        wdrc_filename = f"wdrc_{str(drce_lambda)}.pkl"
    wdrc_filepath = rawpath + wdrc_filename
    
    lqg_filename = f"lqg.pkl"
    lqg_filepath = rawpath + lqg_filename

    with open(drce_filepath, 'rb') as drce_file:
        drce_data = pickle.load(drce_file)
    with open(wdrc_filepath, 'rb') as wdrc_file:
        wdrc_data = pickle.load(wdrc_file)
    with open(lqg_filepath, 'rb') as lqg_file:
        lqg_data = pickle.load(lqg_file)

    

    summarize(lqg_data, wdrc_data, drce_data, args.dist, args.noise_dist,  path , args.num_sim, args.trajectory, plot_results=True)
    

