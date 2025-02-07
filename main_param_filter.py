#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file generates data for Gaussian(Normal) and U-Quadratic distributions
# 4 method implemented (WDRC, WDRC+DRKF, WDRC+DRMMSE, WDR-CE (Ours))

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.WDRC_DRKF import WDRCDRKF
from controllers.DRCE import DRCE
from controllers.DRCMMSE import DRCMMSE

import os
import pickle

def uniform(a, b, N=1):
    n = a.shape[0]
    x = a + (b-a)*np.random.rand(N,n)
    return x.T

def normal(mu, Sigma, N=1):
    x = np.random.multivariate_normal(mu[:,0], Sigma, size=N).T
    return x
def quad_inverse(x, b, a):
    row = x.shape[0]
    col = x.shape[1]
    for i in range(row):
        for j in range(col):
            beta = (a[j]+b[j])/2.0
            alpha = 12.0/((b[j]-a[j])**3)
            tmp = 3*x[i][j]/alpha - (beta - a[j])**3
            if 0<=tmp:
                x[i][j] = beta + ( tmp)**(1./3.)
            else:
                x[i][j] = beta -(-tmp)**(1./3.)
    return x

# quadratic U-shape distrubituon in [wmin , wmax]
def quadratic(wmax, wmin, N=1):
    n = wmin.shape[0]
    x = np.random.rand(N, n)
    #print("wmax : " , wmax)
    x = quad_inverse(x, wmax, wmin)
    return x.T

def multimodal(mu, Sigma, N=1):
    modes = 2
    n = mu[0].shape[0]
    x = np.zeros((n,N,modes))
    for i in range(modes):
        w = np.random.normal(size=(N,n))
        if (Sigma[i] == 0).all():
            x[:,:,i] = mu[i]
        else:
            x[:,:,i] = mu[i] + np.linalg.cholesky(Sigma[i]) @ w.T
    w = 0.5
    y = x[:,:,0]*w + x[:,:,1]*(1-w)
    return y

def gen_sample_dist(dist, T, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist=="normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist=="uniform":
        w = uniform(w_max, w_min, N=N_sample)
    elif dist=="multimodal":
        w = multimodal(mu_w, Sigma_w, N=N_sample)
    elif dist=="quadratic":
        w = quadratic(w_max, w_min, N=N_sample)

    mean_ = np.average(w, axis = 1)
    diff = (w.T - mean_)[...,np.newaxis]
    var_ = np.average( (diff @ np.transpose(diff, (0,2,1))) , axis = 0)
    return np.tile(mean_[...,np.newaxis], (T, 1, 1)), np.tile(var_, (T, 1, 1))

def gen_sample_dist_inf(dist, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist=="normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist=="uniform":
        w = uniform(w_max, w_min, N=N_sample)
    elif dist=="multimodal":
        w = multimodal(mu_w, Sigma_w, N=N_sample)
    elif dist=="quadratic":
        w = quadratic(w_max, w_min, N=N_sample)
        
    mean_ = np.average(w, axis = 1)[...,np.newaxis]
    var_ = np.cov(w)
    return mean_, var_

def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()

def main(dist, noise_dist1, num_sim, num_samples, num_noise_samples, T):
    
    lambda_ = 10
    seed = 2024 # Random seed
    noisedist = [noise_dist1]
    #noisedist = ["normal", "uniform", "quadratic"]
    
    theta_w = 1.0 # will not be used for this file!!!
    
    output_J_WDRC_mean, output_J_WDRC_DRKF_mean, output_J_DRCE_mean, output_J_DRCMMSE_mean =[], [], [], []
    output_J_WDRC_std, output_J_WDRC_DRKF_std, output_J_DRCE_std, output_J_DRCMMSE_std=[], [], [], []
    #-------Initialization-------
    nx = 10 #state dimension
    nu = 10 #control input dimension
    ny = 9#output dimension
    temp = np.ones((nx, nx))
    A = np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2)
    B = Q = R = Qf = np.eye(10)
    #C = np.eye(10)
    C = np.hstack([np.eye(9), np.zeros((9,1))])
    #----------------------------
    # Sample size
    num_x0_samples = 20 #  x0 samples 
    if dist=='normal':
        num_samples=num_noise_samples=15
        num_x0_samples=10
    else:
        num_samples=num_noise_samples=num_x0_samples=20
    num_noise_list = [num_noise_samples]
    #---------------------
    # You can change theta_v list and lambda_list ! but you also need to change lists at plot_params4_F.py to get proper plot

    theta_v_list = [2.0, 4.0, 6.0, 8.0, 10.0] # radius of noise ambiguity set
    theta_w_list = [2.0, 4.0, 6.0, 8.0, 10.0] # radius of noise ambiguity set
    lambda_list = [10, 20, 30, 40, 50] # disturbance distribution penalty parameter

    theta_x0 = 5.0 # radius of state ambiguity set  
    use_lambda = True # If use_lambda is True, we will use lambda_list. If use_lambda is False, we will use theta_w_list
    use_optimal_lambda = False
    if use_lambda:
        dist_parameter_list = lambda_list
    else:
        dist_parameter_list = theta_w_list
    
    
    
    WDRC_lambda = np.zeros((15,15))
    WDRC_DRKF_lambda = np.zeros((15,15))
    WDRC_DRMMSE_lambda = np.zeros((15,15))
    DRCE_lambda = np.zeros((15,15))
    
    for noise_dist in noisedist:
        for idx_w, dist_parameter in enumerate(dist_parameter_list):
            for idx_v, theta in enumerate(theta_v_list):
                for num_noise in num_noise_list:
                    
                    np.random.seed(seed) # fix Random seed!
                    print("--------------------------------------------")
                    print("number of noise sample : ", num_noise)
                    print("number of disturbance sample : ", num_samples)
                    if use_lambda:
                        lambda_ = dist_parameter
                        print("disturbance : ", dist, "/ noise : ", noise_dist, "/ num_noise : ", num_noise, "/ lambda: ", lambda_, "/ theta_v : ", theta)
                    else:
                        theta_w = dist_parameter
                        print("disturbance : ", dist, "/ noise : ", noise_dist, "/ num_noise : ", num_noise, "/ theta_w: ", theta_w, "/ theta_v : ", theta)
                    
                
                    if use_lambda:
                        path = "./results/{}_{}/finite/multiple/params_lambda/filter/".format(dist, noise_dist)
                    else:
                        path = "./results/{}_{}/finite/multiple/params_thetas/filter/".format(dist, noise_dist)
                        
                    if not os.path.exists(path):
                        os.makedirs(path)
                
                    #-------Disturbance Distribution-------
                    if dist == "normal":
                        #disturbance distribution parameters
                        w_max = None
                        w_min = None
                        mu_w = 0.2*np.ones((nx, 1))
                        Sigma_w= 0.1*np.eye(nx)
                        #initial state distribution parameters
                        x0_max = None
                        x0_min = None
                        x0_mean = 1*np.ones((nx,1))
                        #x0_mean[-1]=-1
                        x0_cov = 0.1*np.eye(nx)
                    elif dist == "quadratic":
                        #disturbance distribution parameters
                        w_max = 1.0*np.ones(nx)
                        w_min = -0.5*np.ones(nx)
                        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
                        Sigma_w = 3.0/20.0*np.diag((w_max - w_min)**2)
                        #initial state distribution parameters
                        x0_max = 0.5*np.ones(nx)
                        x0_min = 0.0*np.ones(nx)
                        #x0_max[-1] = 0.0
                        #x0_min[-1] = -2.0
                        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
                        x0_cov = 3.0/20.0 *np.diag((x0_max - x0_min)**2)
                        
                    #-------Noise distribution ---------#
                    if noise_dist =="normal":
                        v_max = None
                        v_min = None
                        M = 1.5*np.eye(ny) #observation noise covariance
                        mu_v = 0.2*np.ones((ny, 1))
                    elif noise_dist =="quadratic":
                        v_min = -1.5*np.ones(ny)
                        v_max = 3.0*np.ones(ny)
                        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                        M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance
                        
                        
                    #-------Estimate the nominal distribution-------
                
                    # Nominal initial state distribution
                    x0_mean_hat, x0_cov_hat = gen_sample_dist(dist, 1, num_x0_samples, mu_w=x0_mean, Sigma_w=x0_cov, w_max=x0_max, w_min=x0_min)
                    # Nominal Disturbance distribution
                    mu_hat, Sigma_hat = gen_sample_dist(dist, T+1, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
                    # Nominal Noise distribution
                    v_mean_hat, M_hat = gen_sample_dist(noise_dist, T+1, num_noise, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min)
                    
                    M_hat = M_hat + 1e-6*np.eye(ny) # to prevent numerical error from inverse in standard KF at small sample size
                    
                    #-------Create a random system-------
                    system_data = (A, B, C, Q, Qf, R, M)
                    
                    #-------Perform n independent simulations and summarize the results-------
                    
                    output_wdrc_list = []
                    output_wdrc_drkf_list = []
                    output_drce_list = []
                    output_drcmmse_list = []
                    
                    #Initialize controllers
                    if use_optimal_lambda == True:
                        lambda_ = WDRC_lambda[idx_w][idx_v]
                    wdrc = WDRC(lambda_, theta_w, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda, use_optimal_lambda)
                    if use_optimal_lambda == True:
                        lambda_ = DRCE_lambda[idx_w][idx_v]
                    drce = DRCE(lambda_, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat,  M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda, use_optimal_lambda)
                    if use_optimal_lambda == True:
                        lambda_ = WDRC_DRKF_lambda[idx_w][idx_v]
                    wdrcdrkf = WDRCDRKF(lambda_, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat,  M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda, use_optimal_lambda)
                    if use_optimal_lambda == True:
                        lambda_ = WDRC_DRMMSE_lambda[idx_w][idx_v]
                    drcmmse = DRCMMSE(lambda_, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat,  M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda, use_optimal_lambda)
                    
                    wdrcdrkf.backward()
                    drcmmse.backward()
                    wdrc.backward()
                    drce.backward()
                        
                    print('---------------------')
                    # Save the optimzed lambda
                    WDRC_lambda[idx_w][idx_v] = wdrc.lambda_
                    DRCE_lambda[idx_w][idx_v] = drce.lambda_
                    WDRC_DRKF_lambda[idx_w][idx_v] = wdrcdrkf.lambda_
                    WDRC_DRMMSE_lambda[idx_w][idx_v] = drcmmse.lambda_

                    
                    #----------------------------
                    print("Running DRCE Forward step ...")
                    for i in range(num_sim):
                        
                        #Perform state estimation and apply the controller
                        output_drce = drce.forward()
                        output_drce_list.append(output_drce)
                        if i%50==0:
                            print("Simulation #",i, ' | cost (DRCE):', output_drce['cost'][0], 'time (DRCE):', output_drce['comp_time'])
                    J_DRCE_list = []
                    for out in output_drce_list:
                        J_DRCE_list.append(out['cost'])
                    J_DRCE_mean= np.mean(J_DRCE_list, axis=0)
                    J_DRCE_std = np.std(J_DRCE_list, axis=0)
                    output_J_DRCE_mean.append(J_DRCE_mean[0])
                    output_J_DRCE_std.append(J_DRCE_std[0])
                    print(" Average cost (DRCE) : ", J_DRCE_mean[0])
                    print(" std (DRCE) : ", J_DRCE_std[0])
                    
                    #----------------------------
                    print("Running DRCMMSE Forward step ...")
                    for i in range(num_sim):
                        
                        #Perform state estimation and apply the controller
                        output_drcmmse = drcmmse.forward()
                        output_drcmmse_list.append(output_drcmmse)
                        if i%50==0:
                            print("Simulation #",i, ' | cost (DRCMMSE):', output_drcmmse['cost'][0], 'time (DRCMMSE):', output_drcmmse['comp_time'])
                    
                    J_DRCMMSE_list = []
                    for out in output_drcmmse_list:
                        J_DRCMMSE_list.append(out['cost'])
                    J_DRCMMSE_mean= np.mean(J_DRCMMSE_list, axis=0)
                    J_DRCMMSE_std = np.std(J_DRCMMSE_list, axis=0)
                    output_J_DRCMMSE_mean.append(J_DRCMMSE_mean[0])
                    output_J_DRCMMSE_std.append(J_DRCMMSE_std[0])
                    print(" Average cost (DRCMMSE) : ", J_DRCMMSE_mean[0])
                    print(" std (DRCMMSE) : ", J_DRCMMSE_std[0])
                    
                    #----------------------------             
                    np.random.seed(seed) # fix Random seed!
                    print("Running WDRC Forward step ...")  
                    for i in range(num_sim):
                
                        #Perform state estimation and apply the controller
                        output_wdrc = wdrc.forward()
                        output_wdrc_list.append(output_wdrc)
                        if i%50==0:
                            print("Simulation #",i, ' | cost (WDRC):', output_wdrc['cost'][0], 'time (WDRC):', output_wdrc['comp_time'])
                    
                    J_WDRC_list = []
                    for out in output_wdrc_list:
                        J_WDRC_list.append(out['cost'])
                    J_WDRC_mean= np.mean(J_WDRC_list, axis=0)
                    J_WDRC_std = np.std(J_WDRC_list, axis=0)
                    output_J_WDRC_mean.append(J_WDRC_mean[0])
                    output_J_WDRC_std.append(J_WDRC_std[0])
                    print(" Average cost (WDRC) : ", J_WDRC_mean[0])
                    print(" std (WDRC) : ", J_WDRC_std[0])
                    #----------------------------             
                    np.random.seed(seed) # fix Random seed!
                    print("Running WDRCDRKF Forward step ...")  
                    for i in range(num_sim):
                
                        #Perform state estimation and apply the controller
                        output_wdrc_drkf = wdrcdrkf.forward()
                        output_wdrc_drkf_list.append(output_wdrc_drkf)
                        if i%50==0:
                            print("Simulation #",i, ' | cost (WDRC_DRKF):', output_wdrc_drkf['cost'][0], 'time (WDRC_DRKF):', output_wdrc_drkf['comp_time'])
                    
                    J_WDRC_DRKF_list = []
                    for out in output_wdrc_drkf_list:
                        J_WDRC_DRKF_list.append(out['cost'])
                    J_WDRC_DRKF_mean= np.mean(J_WDRC_DRKF_list, axis=0)
                    J_WDRC_DRKF_std = np.std(J_WDRC_DRKF_list, axis=0)
                    output_J_WDRC_DRKF_mean.append(J_WDRC_DRKF_mean[0])
                    output_J_WDRC_DRKF_std.append(J_WDRC_DRKF_std[0])
                    print(" Average cost (WDRC_DRKF) : ", J_WDRC_DRKF_mean[0])
                    print(" std (WDRC_DRKF) : ", J_WDRC_DRKF_std[0])
                    
                    # -------------------------
                    # Collect State Estimation errors
                    J_MSE_WDRC_DRKF_list,J_MSE_WDRC_list,J_MSE_DRCE_list,J_MSE_DRCMMSE_list = [],[],[],[]
                   
                    for out in output_wdrc_list:
                        J_MSE_WDRC_list.append(out['mse'])
                    for out in output_wdrc_drkf_list:
                        J_MSE_WDRC_DRKF_list.append(out['mse'])
                    for out in output_drce_list:
                        J_MSE_DRCE_list.append(out['mse'])
                    for out in output_drcmmse_list:
                        J_MSE_DRCMMSE_list.append(out['mse'])
                        
                    #J_MSE_LQG_mean = np.mean(J_MSE_LQG_list)
                    J_MSE_WDRC_mean = np.mean(J_MSE_WDRC_list)
                    J_MSE_WDRC_DRKF_mean = np.mean(J_MSE_WDRC_DRKF_list)
                    J_MSE_DRCE_mean = np.mean(J_MSE_DRCE_list)
                    J_MSE_DRCMMSE_mean = np.mean(J_MSE_DRCMMSE_list)
                    
                    #-----------------------------------------
                    # Save data #
                    theta_v_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
                    theta_w_ = f"_{str(theta_w).replace('.', '_')}" # change 1.0 to 1_0 for file name
                    if use_lambda:
                        save_data(path + 'drce_' + str(lambda_) + 'and' + theta_v_+ '.pkl', J_DRCE_mean)
                        save_data(path + 'drcmmse_' + str(lambda_) + 'and' + theta_v_+ '.pkl', J_DRCMMSE_mean)
                        save_data(path + 'wdrc_drkf' + str(lambda_) + 'and' + theta_v_+ '.pkl', J_WDRC_DRKF_mean)
                        save_data(path + 'wdrc_' + str(lambda_) + '.pkl', J_WDRC_mean)
                        
                        save_data(path + 'drce_mse_' + str(lambda_) + 'and' + theta_v_+ '.pkl', J_MSE_DRCE_mean)
                        save_data(path + 'drcmmse_mse_' + str(lambda_) + 'and' + theta_v_+ '.pkl', J_MSE_DRCMMSE_mean)
                        save_data(path + 'wdrc_drkf_mse_' + str(lambda_) + 'and' + theta_v_+ '.pkl', J_MSE_WDRC_DRKF_mean)
                        save_data(path + 'wdrc_mse_' + str(lambda_) + '.pkl', J_MSE_WDRC_mean)
                        
                    else:
                        save_data(path + 'drcmmse' + theta_w_ + 'and' + theta_v_+ '.pkl', J_DRCMMSE_mean)
                        save_data(path + 'drce' + theta_w_ + 'and' + theta_v_+ '.pkl', J_DRCE_mean)
                        save_data(path + 'wdrc_drkf' + theta_w_ + 'and' + theta_v_+ '.pkl', J_WDRC_DRKF_mean)
                        save_data(path + 'wdrc' + theta_w_ + '.pkl', J_WDRC_mean)
                        
                        save_data(path + 'drcmmse_mse' + theta_w_ + 'and' + theta_v_+ '.pkl', J_MSE_DRCMMSE_mean)
                        save_data(path + 'wdrc_drkf_mse' + theta_w_ + 'and' + theta_v_+ '.pkl', J_MSE_WDRC_DRKF_mean)
                        save_data(path + 'drce_mse' + theta_w_ + 'and' + theta_v_+ '.pkl', J_MSE_DRCE_mean)
                        save_data(path + 'wdrc_mse' + theta_w_ + '.pkl', J_MSE_WDRC_mean)
                        
                        
                    save_data(path + 'filter_wdrc_lambda.pkl',WDRC_lambda)
                    save_data(path + 'filter_wdrc_drkf_lambda.pkl',WDRC_DRKF_lambda)
                    save_data(path + 'filter_wdrc_drmmse_lambda.pkl',WDRC_DRMMSE_lambda)
                    save_data(path + 'filter_drce_lambda.pkl',DRCE_lambda)
            
                    #Summarize and plot the results
                    print('\n-------Summary-------')
                    print("dist : ", dist,"/ noise dist : ", noise_dist, "/ num_samples : ", num_samples, "/ num_noise_samples : ", num_noise, "/seed : ", seed)
                    
    print("Params data generation Completed !")
    print("Please make sure your lambda_list(or theta_w_list) and theta_v_list in plot_params4_F.py is as desired")
    
    if use_lambda:
        print("Now use : python plot_params4_F.py --use_lambda --dist "+ dist + " --noise_dist " + noise_dist)
    else:
        print("Now use : python plot_params4_F.py --dist "+ dist + " --noise_dist " + noise_dist)
    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or quadratic)
    parser.add_argument('--num_sim', required=False, default=500, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=20, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=20, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=20, type=int) #horizon length
    
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon)
