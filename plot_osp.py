#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def summarize_theta_w(J_DRCE_mean_all_samp, J_DRCE_std_all_samp, DRCE_prob_all_samp, theta_list, num_noise_list, dist, noise_dist):
    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    })
    
    fig = plt.figure(figsize=(6,4), dpi=300)
    ax = fig.gca()
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, num_noise in enumerate(num_noise_list):
        plt.plot(theta_list, J_DRCE_mean_all_samp[i], color=colors[i], marker='.', markersize=7, label=rf'$N={num_noise}$')
        plt.fill_between(theta_list, J_DRCE_mean_all_samp[i] + 0.25*J_DRCE_std_all_samp[i], J_DRCE_mean_all_samp[i] - 0.25*J_DRCE_std_all_samp[i], facecolor=colors[i], alpha=0.3)
        
    #plt.xscale('log')
    #plt.yscale('log')
    
    
    ax.set_xlabel(r'$\theta$', fontsize=30)
    ax.set_ylabel(r'Out-of-Sample Cost', fontsize=30)
    legend = ax.legend(
        bbox_to_anchor=(1.02, 1.03),  # Moves legend further out of the plot
        loc='upper right',
        frameon=True,
        framealpha=1.0,
        facecolor='white',
        fontsize=24,  # Reduced fontsize to make legend more compact
        borderpad=0.3,
        handletextpad=0.2,  # Reduce space between legend handle and text
        labelspacing=0.1    # Reduce vertical space between entries
    )
    ax.grid()
    ax.set_xlim([theta_list[0], theta_list[-1]])

    y_min = np.min([J_DRCE_mean_all_samp[i] - 0.25*J_DRCE_std_all_samp[i] for i in range(len(num_noise_list))]) 
    y_max = np.max([J_DRCE_mean_all_samp[i] + 0.25*J_DRCE_std_all_samp[i] for i in range(len(num_noise_list))]) 
    
    #x_ticks = np.linspace(theta_list[0], theta_list[-1], num=5)  # You can adjust the number of ticks with `num`
    # Set the z-ticks on the plot
    
    #ax.set_xticks(x_ticks)

    y_ticks = np.linspace(int(np.floor(y_min)), int(np.ceil(y_max)), num=5)  # You can adjust the number of ticks with `num`
    y_ticks = [int(tick) for tick in y_ticks]
    # Set the z-ticks on the plot
    ax.set_yticks(y_ticks)
    ax.set_xticks([2, 4, 6, 8, 10])

    ax.tick_params(axis='x', which='major', labelsize=24, pad=5)  # Add padding between z ticks and axis
    ax.tick_params(axis='y', which='major', labelsize=24, pad=5)  # Add padding between z ticks and axis
    
    
    plt.savefig(path +f'/OSP_{dist}_{noise_dist}.pdf', dpi=300, bbox_inches="tight")
    plt.clf()

    
    fig = plt.figure(figsize=(6,4), dpi=300)
    ax = fig.gca()
    
    for i, num_noise in enumerate(num_noise_list):
        plt.plot(theta_list, DRCE_prob_all_samp[i], color=colors[i], marker='.', markersize=7, label=rf'$N={num_noise}$')
      

    
    ax.set_xlabel(r'$\theta$', fontsize=30)
    ax.set_ylabel(r'Reliability', fontsize=30)
    legend = ax.legend(
        bbox_to_anchor=(1.02, 0.45),  # Moves legend further out of the plot
        loc='upper right',
        frameon=True,
        framealpha=1.0,
        facecolor='white',
        fontsize=24,  # Reduced fontsize to make legend more compact
        borderpad=0.3,
        handletextpad=0.2,  # Reduce space between legend handle and text
        labelspacing=0.1    # Reduce vertical space between entries
    )
    ax.grid()
    ax.set_xlim([theta_list[0], theta_list[-1]])

    y_min = np.min([DRCE_prob_all_samp[i] for i in range(len(num_noise_list))]) 
    y_max = np.max([DRCE_prob_all_samp[i] for i in range(len(num_noise_list))]) 
    
    #x_ticks = np.linspace(theta_list[0], theta_list[-1], step=5)  # You can adjust the number of ticks with `num`
    # Set the z-ticks on the plot
    
    ax.set_xticks([2, 4, 6, 8, 10])

    y_ticks = np.linspace(int(np.floor(y_min)), int(np.ceil(y_max)), num=5)  # You can adjust the number of ticks with `num`
    #y_ticks = [int(tick) for tick in y_ticks]
    # Set the z-ticks on the plot
    ax.set_yticks(y_ticks)
    
    ax.tick_params(axis='x', which='major', labelsize=24, pad=5)  # Add padding between z ticks and axis
    ax.tick_params(axis='y', which='major', labelsize=24, pad=5)  # Add padding between z ticks and axis

    plt.savefig(path + f'/OSP_Prob_{dist}_{noise_dist}.pdf', dpi=300, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    args = parser.parse_args()
    
    theta_list = [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] # radius of noise ambiguity set
    
    num_noise_list = [10, 15, 20] #
    
    noisedist = [args.noise_dist]
    
    J_DRCE_mean_all_samp = []
    J_DRCE_std_all_samp = []
    DRCE_prob_all_samp = []
            
    for noise_dist in noisedist:
        for idx, num_noise in enumerate(num_noise_list):
            J_DRCE_mean_samp = []
            J_DRCE_std_samp = []
            DRCE_prob_samp = []
            
            for theta in theta_list:
                path = "./results/{}_{}/finite/multiple/OS/".format(args.dist, args.noise_dist)

                theta_v_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
                theta_w_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
                
                drce_file = open(path + 'N=' + str(num_noise) + '/drce_mean_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', 'rb')
                output_J_DRCE_mean = pickle.load(drce_file)
                drce_file.close()
                drce_file = open(path + 'N=' + str(num_noise) + '/drce_std_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', 'rb')
                output_J_DRCE_std = pickle.load(drce_file)
                drce_file.close()
                drce_file = open(path + 'N=' + str(num_noise) + '/drce_prob_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', 'rb')
                output_DRCE_prob = pickle.load(drce_file)
                drce_file.close()
                
                
                J_DRCE_mean_samp.append(output_J_DRCE_mean[-1])
                J_DRCE_std_samp.append(output_J_DRCE_std[-1])
                
                DRCE_prob_samp.append(output_DRCE_prob[-1])
                
            J_DRCE_mean_all_samp.append(J_DRCE_mean_samp)
            J_DRCE_std_all_samp.append(J_DRCE_std_samp)
            DRCE_prob_all_samp.append(DRCE_prob_samp)                

    
    J_DRCE_mean_all_samp = np.array(J_DRCE_mean_all_samp)
    J_DRCE_std_all_samp = np.array(J_DRCE_std_all_samp)
    DRCE_prob_all_samp = np.array(DRCE_prob_all_samp)

    summarize_theta_w(J_DRCE_mean_all_samp, J_DRCE_std_all_samp, DRCE_prob_all_samp, theta_list, num_noise_list, args.dist, args.noise_dist)

