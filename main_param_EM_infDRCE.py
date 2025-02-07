#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file generates data for experiments comparing three controllers:
   - LQG,
   - Finite–horizon DRCE, and 
   - Infinite–horizon DRCE.
   
It uses your disturbance/noise distributions (normal or quadratic) and
properly sets the sizes of the nominal means and covariances for the two horizon settings.
"""

import numpy as np
import argparse
from controllers.inf_DRCE import inf_DRCE
from controllers.LQG import LQG
from controllers.DRCE import DRCE
from numpy.linalg import norm
from joblib import Parallel, delayed
from pykalman import KalmanFilter
import os
import pickle

reg_eps = 1e-6

def uniform(a, b, N=1):
    n = a.shape[0]
    x = a + (b - a) * np.random.rand(N, n)
    return x.T

def normal(mu, Sigma, N=1):
    x = np.random.multivariate_normal(mu[:, 0], Sigma, size=N).T
    return x

def quad_inverse(x, b, a):
    row, col = x.shape
    for i in range(row):
        for j in range(col):
            beta = (a[j] + b[j]) / 2.0
            alpha = 12.0 / ((b[j] - a[j]) ** 3)
            tmp = 3 * x[i][j] / alpha - (beta - a[j]) ** 3
            if tmp >= 0:
                x[i][j] = beta + (tmp) ** (1. / 3.)
            else:
                x[i][j] = beta - (-tmp) ** (1. / 3.)
    return x

# quadratic U-shape distribution in [wmin, wmax]
def quadratic(wmax, wmin, N=1):
    n = wmin.shape[0]
    x = np.random.rand(N, n)
    x = quad_inverse(x, wmax, wmin)
    return x.T

def multimodal(mu, Sigma, N=1):
    modes = 2
    n = mu[0].shape[0]
    x = np.zeros((n, N, modes))
    for i in range(modes):
        w = np.random.normal(size=(N, n))
        if (Sigma[i] == 0).all():
            x[:, :, i] = mu[i]
        else:
            x[:, :, i] = mu[i] + np.linalg.cholesky(Sigma[i]) @ w.T
    w = 0.5
    y = x[:, :, 0] * w + x[:, :, 1] * (1 - w)
    return y

def gen_sample_dist(dist, T, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist == "normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist == "uniform":
        w = uniform(w_max, w_min, N=N_sample)
    elif dist == "multimodal":
        w = multimodal(mu_w, Sigma_w, N=N_sample)
    elif dist == "quadratic":
        w = quadratic(w_max, w_min, N=N_sample)

    mean_ = np.average(w, axis=1)
    diff = (w.T - mean_)[..., np.newaxis]
    var_ = np.average((diff @ np.transpose(diff, (0, 2, 1))), axis=0)
    return np.tile(mean_[..., np.newaxis], (T, 1, 1)), np.tile(var_, (T, 1, 1))

def gen_sample_dist_inf(dist, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist == "normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist == "uniform":
        w = uniform(w_max, w_min, N=N_sample)
    elif dist == "multimodal":
        w = multimodal(mu_w, Sigma_w, N=N_sample)
    elif dist == "quadratic":
        w = quadratic(w_max, w_min, N=N_sample)
        
    mean_ = np.average(w, axis=1)[..., np.newaxis]
    var_ = np.cov(w)
    return mean_, var_

def save_pickle_data(path, data):
    with open(path, 'wb') as output:
        pickle.dump(data, output)

def load_pickle_data(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)
    
def save_data(path, data):
    with open(path, 'wb') as output:
        pickle.dump(data, output)

# Function to generate the true states and measurements (data generation)
def generate_data(T, nx, ny, nu, A, B, C, mu_w, Sigma_w, mu_v, M,
                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist):
    u = np.zeros((T, nu, 1))
    x_true_all = np.zeros((T + 1, nx, 1))
    y_all = np.zeros((T, ny, 1))
    
    # Initialize the true state
    if dist == "normal":
        x_true = normal(x0_mean, x0_cov)
    elif dist == "quadratic":
        x_true = quadratic(x0_max, x0_min)
    
    x_true_all[0] = x_true
    
    for t in range(T):
        if dist == "normal":
            true_w = normal(mu_w, Sigma_w)
            true_v = normal(mu_v, M)
        elif dist == "quadratic":
            true_w = quadratic(w_max, w_min)
            true_v = quadratic(v_max, v_min)
        
        y_t = C @ x_true + true_v
        y_all[t] = y_t
        x_true = A @ x_true + B @ u[t] + true_w
        x_true_all[t + 1] = x_true

    return x_true_all, y_all

def main(dist, noise_dist, num_sim, num_samples, num_noise_samples, T, infinite):
    lambda_ = 10
    seed = 2024
    np.random.seed(seed)
    tol = 1e-2
    num_noise_list = [num_noise_samples]
    
    # LQG, finite horizon DRCE, and infinite horizon DRCE.
    output_J_LQG_mean = []
    output_J_LQG_std = []
    output_J_DRCE_mean = []       # finite horizon DRCE
    output_J_DRCE_std = []
    output_J_inf_DRCE_mean = []   # infinite horizon DRCE
    output_J_inf_DRCE_std = []
    
    # ------- System Initialization -------
    nx = 10  # state dimension
    nu = 10  # control input dimension
    ny = 10  # output dimension
    temp = np.ones((nx, nx))
    A = 0.2 * (np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2))
    B = np.eye(nx)
    C = Q = R = Qf = np.eye(nx)
    # ----------------------------
    if dist == 'normal':
        theta_v_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # theta_w_list = [0.1, 0.2, 0.5, 1.0, 1.5]
        #theta_v_list = [1.0]
        theta_w_list = [0.1]
    else:
        theta_v_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        theta_w_list = [0.1, 0.2, 0.5, 1.0, 1.5]
    lambda_list = [ 5, 10, 15, 20, 25, 30, 35, 40]
    
    theta_w_local = 1.0
    theta_x0 = 0.5
    use_lambda = True
    use_optimal_lambda = False
    if use_lambda:
        dist_parameter_list = lambda_list
    else:
        dist_parameter_list = theta_w_list

    # --- Load optimal lambda for DRCE (if available) ---
    if dist == "normal":
        DRCE_lambda_file = open('./inputs/io_nn/nonzero_drce_lambda.pkl', 'rb')
    elif dist == "quadratic":
        DRCE_lambda_file = open('./inputs/io_qq/nonzero_drce_lambda.pkl', 'rb')
    DRCE_lambda = pickle.load(DRCE_lambda_file)
    DRCE_lambda_file.close()
    
    print("DRCE_lambda:")
    print(DRCE_lambda)
    # ------- Disturbance Distribution -------
    if dist == "normal":
        w_max = None
        w_min = None
        mu_w = -0.3 * np.ones((nx, 1))
        Sigma_w = 0.3 * np.eye(nx)
        x0_max = None
        x0_min = None
        x0_mean = 0.1 * np.ones((nx, 1))
        x0_cov = 0.1 * np.eye(nx)
    elif dist == "quadratic":
        w_max = 1.0 * np.ones(nx)
        w_min = -2.0 * np.ones(nx)
        mu_w = (0.5 * (w_max + w_min))[..., np.newaxis]
        Sigma_w = 3.0 / 20.0 * np.diag((w_max - w_min) ** 2)
        x0_max = 0.21 * np.ones(nx)
        x0_min = 0.19 * np.ones(nx)
        x0_mean = (0.5 * (x0_max + x0_min))[..., np.newaxis]
        x0_cov = 3.0 / 20.0 * np.diag((x0_max - x0_min) ** 2)
    # ------- Noise Distribution ---------
    if noise_dist == "normal":
        v_max = None
        v_min = None
        M = 1.0 * np.eye(ny)
        mu_v = 0.5 * np.ones((ny, 1))
    elif noise_dist == "quadratic":
        v_min = -1.0 * np.ones(ny)
        v_max = 2.0 * np.ones(ny)
        mu_v = (0.5 * (v_max + v_min))[..., np.newaxis]
        M = 3.0 / 20.0 * np.diag((v_max - v_min) ** 2)
    
    print(f'Real data:\n mu_w: {mu_w},\n mu_v: {mu_v},\n Sigma_w: {Sigma_w},\n Sigma_v: {M}')
    
    N = 50
    x_all, y_all = generate_data(N, nx, ny, nu, A, B, C, mu_w, Sigma_w, mu_v, M,
                                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist)
    
    y_all = y_all.squeeze()
    eps_param = 1e-5
    eps_log = 1e-4
    
    # --- Initialize nominal parameters for estimation using EM with a Kalman Filter ---
    mu_w_hat = np.zeros(nx)
    mu_v_hat = np.zeros(ny)
    mu_x0_hat = x0_mean.squeeze()
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    Sigma_x0_hat = x0_cov
    
    kf = KalmanFilter(A, C, Sigma_w_hat, Sigma_v_hat, mu_w_hat, mu_v_hat, mu_x0_hat, Sigma_x0_hat,
                      em_vars=[
                          'transition_covariance', 'observation_covariance',
                          'transition_offsets', 'observation_offsets',
                      ])
    
    max_iter = 50
    loglikelihoods = np.zeros(max_iter)
    errors_mu_w = []
    errors_mu_v = []
    errors_mu_x0 = []
    errors_Sigma_w = []
    errors_Sigma_v = []
    errors_Sigma_x0 = []
    
    for i in range(max_iter):
        print(f'------- EM Iteration {i} / {max_iter} ------------')
        kf = kf.em(X=y_all, n_iter=1)
        loglikelihoods[i] = kf.loglikelihood(y_all)
        
        Sigma_w_hat = kf.transition_covariance
        Sigma_v_hat = kf.observation_covariance
        mu_w_hat = kf.transition_offsets
        mu_v_hat = kf.observation_offsets
        mu_x0_hat = kf.initial_state_mean
        Sigma_x0_hat = kf.initial_state_covariance

        error_mu_w = norm(mu_w_hat - mu_w)
        error_mu_v = norm(mu_v_hat - mu_v)
        error_mu_x0 = norm(mu_x0_hat - x0_mean)
        error_Sigma_w = norm(Sigma_w_hat - Sigma_w, 'fro')
        error_Sigma_v = norm(Sigma_v_hat - M, 'fro')
        error_Sigma_x0 = norm(Sigma_x0_hat - x0_cov, 'fro')
        
        errors_mu_w.append(error_mu_w)
        errors_mu_v.append(error_mu_v)
        errors_mu_x0.append(error_mu_x0)
        errors_Sigma_w.append(error_Sigma_w)
        errors_Sigma_v.append(error_Sigma_v)
        errors_Sigma_x0.append(error_Sigma_x0)
        
        params_conv = (error_mu_w <= eps_param and error_mu_v <= eps_param and
                       error_mu_x0 <= eps_param and error_Sigma_w <= eps_param and
                       error_Sigma_v <= eps_param and error_Sigma_x0 <= eps_param)
        
        if i > 0:
            if loglikelihoods[i] - loglikelihoods[i-1] <= eps_log and params_conv:
                print('Converged!')
                break

    print("Nominal distributions are ready")
    
    # Reshape estimated nominal parameters
    mu_w_hat = np.array(mu_w_hat).reshape(-1, 1)
    mu_v_hat = np.array(mu_v_hat).reshape(-1, 1)
    M_hat = Sigma_v_hat

    print("Estimated mu_w:")
    print(mu_w_hat)
    print("\nTrue mu_w:")
    print(mu_w)
    print("\nEstimation Error (mu_w): {:.6f}".format(error_mu_w))

    print("\nEstimated Sigma_w:")
    print(Sigma_w_hat)
    print("\nTrue Sigma_w:")
    print(Sigma_w)
    print("\nEstimation Error (Sigma_w): {:.6f}".format(error_Sigma_w))

    print("\nEstimated mu_v:")
    print(mu_v_hat)
    print("\nTrue mu_v:")
    print(mu_v)

    print("\nEstimated M (Sigma_v):")
    print(M_hat)
    print("\nTrue M:")
    print(M)
    
    # --- Prepare nominal parameters for controllers ---
    # For finite-horizon controllers, tile the nominal parameters over time.
    mu_w_hat_fh = np.tile(mu_w_hat, (T, 1, 1))
    mu_v_hat_fh = np.tile(mu_v_hat, (T+1, 1, 1))
    Sigma_w_hat_fh = np.tile(Sigma_w_hat, (T, 1, 1))
    M_hat_fh = np.tile(M_hat, (T+1, 1, 1))
    
    x0_mean_hat = x0_mean
    x0_cov_hat = x0_cov

    # Create paths for saving results
    temp_results_path = "./temp_results/"
    if not os.path.exists(temp_results_path):
        os.makedirs(temp_results_path)
        
    # Modified simulation function: LQG, finite-horizon DRCE, and infinite-horizon DRCE.
    def perform_simulation(lambda_, noise_dist, dist_parameter, theta, idx_w, idx_v):
        for num_noise in num_noise_list:
            np.random.seed(seed)
            if use_lambda:
                lambda_ = dist_parameter
                theta_w_local = 1.0 # will not be used
            else:
                theta_w_local = dist_parameter  # local copy of theta_w
            
            # Set a results directory
            if use_lambda:
                path = "./results/{}_{}/experiment/params_lambda/".format(dist, noise_dist)
            else:
                path = "./results/{}_{}/experiment/params_thetas/".format(dist, noise_dist)
            if not os.path.exists(path):
                os.makedirs(path)
            
            # ------- Create system data -------
            system_data = (A, B, C, Q, Qf, R, M)
            
            # ------- Instantiate controllers -------
            # Finite-horizon LQG and DRCE use the tiled (finite-horizon) nominal parameters.
            lqg = LQG(T, dist, noise_dist, system_data, mu_w_hat_fh, Sigma_w_hat_fh,
                      x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                      v_max, v_min, mu_v, mu_v_hat_fh, M_hat_fh, x0_mean_hat, x0_cov_hat)
            
            drce = DRCE(lambda_, theta_w_local, theta, theta_x0, T, dist, noise_dist,
                        system_data, mu_w_hat_fh, Sigma_w_hat_fh,
                        x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                        v_max, v_min, mu_v, mu_v_hat_fh, M_hat_fh, x0_mean_hat, x0_cov_hat,
                        use_lambda, use_optimal_lambda)
            
            # Infinite-horizon DRCE uses the original nominal parameters.
            inf_drce = inf_DRCE(lambda_, theta_w_local, theta, theta_x0, T, dist, noise_dist,
                                 system_data, mu_w_hat, Sigma_w_hat,
                                 x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                                 v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat,
                                 use_lambda, use_optimal_lambda)
            
            # --- Solve/backward pass---
            drce.backward()
            inf_drce.backward()
            lqg.backward()
            
            # Lists to collect forward simulation outputs
            output_lqg_list = []
            output_drce_list = []
            output_inf_drce_list = []
            
            # --- Finite-horizon DRCE Forward simulation ---
            print("Running finite-horizon DRCE Forward step ...")
            for i in range(num_sim):
                output_drce = drce.forward()
                output_drce_list.append(output_drce)
                if i % 50 == 0:
                    print("Simulation #", i, '| cost (finite DRCE):', output_drce['cost'][0],'time:', output_drce['comp_time'])
            
            J_DRCE_list = [out['cost'] for out in output_drce_list]
            J_DRCE_mean = np.mean(J_DRCE_list, axis=0)
            J_DRCE_std = np.std(J_DRCE_list, axis=0)
            output_J_DRCE_mean.append(J_DRCE_mean[0])
            output_J_DRCE_std.append(J_DRCE_std[0])
            print(" Average cost (finite DRCE): ", J_DRCE_mean[0])
            print(" std (finite DRCE): ", J_DRCE_std[0])
            
            # --- Infinite-horizon DRCE Forward simulation ---
            print("Running infinite-horizon DRCE Forward step ...")
            for i in range(num_sim):
                output_inf_drce = inf_drce.forward()
                output_inf_drce_list.append(output_inf_drce)
                if i % 50 == 0:
                    print("Simulation #", i, '| cost (infinite DRCE):', output_inf_drce['cost'][0],'time:', output_inf_drce['comp_time'])
            
            J_inf_DRCE_list = [out['cost'] for out in output_inf_drce_list]
            J_inf_DRCE_mean = np.mean(J_inf_DRCE_list, axis=0)
            J_inf_DRCE_std = np.std(J_inf_DRCE_list, axis=0)
            output_J_inf_DRCE_mean.append(J_inf_DRCE_mean[0])
            output_J_inf_DRCE_std.append(J_inf_DRCE_std[0])
            print(" Average cost (infinite DRCE): ", J_inf_DRCE_mean[0])
            print(" std (infinite DRCE): ", J_inf_DRCE_std[0])
            
            # --- LQG Forward simulation ---
            print("Running LQG Forward step ...")
            for i in range(num_sim):
                output_lqg = lqg.forward()
                output_lqg_list.append(output_lqg)
                if i % 50 == 0:
                    print("Simulation #", i, '| cost (LQG):', output_lqg['cost'][0],'time:', output_lqg['comp_time'])
            
            J_LQG_list = [out['cost'] for out in output_lqg_list]
            J_LQG_mean = np.mean(J_LQG_list, axis=0)
            J_LQG_std = np.std(J_LQG_list, axis=0)
            output_J_LQG_mean.append(J_LQG_mean[0])
            output_J_LQG_std.append(J_LQG_std[0])
            print(" Average cost (LQG): ", J_LQG_mean[0])
            print(" std (LQG): ", J_LQG_std[0])
            
            # --- Save cost data ---
            theta_v_str = f"_{str(theta).replace('.', '_')}"  # for file naming
            theta_w_str = f"_{str(theta_w_local).replace('.', '_')}"
            
            # Save cost data for each method
            if use_lambda:
                save_data(path + 'drce_finite_' + str(lambda_) + 'and' + theta_v_str + '.pkl', J_DRCE_mean)
                save_data(path + 'drce_infinite_' + str(lambda_) + 'and' + theta_v_str + '.pkl', J_inf_DRCE_mean)
            else:
                save_data(path + 'drce_finite' + theta_w_str + 'and' + theta_v_str + '.pkl', J_DRCE_mean)
                save_data(path + 'drce_infinite' + theta_w_str + 'and' + theta_v_str + '.pkl', J_inf_DRCE_mean)
            save_data(path + 'lqg.pkl', J_LQG_mean)
            
            # --- Save all raw data ---
            rawpath = os.path.join(path, "raw")
            if not os.path.exists(rawpath):
                os.makedirs(rawpath)
            if use_lambda:
                save_data(os.path.join(rawpath, 'drce_finite_' + str(lambda_) + 'and' + theta_v_str + '.pkl'), output_drce_list)
                save_data(os.path.join(rawpath, 'drce_infinite_' + str(lambda_) + 'and' + theta_v_str + '.pkl'), output_inf_drce_list)
            else:
                save_data(os.path.join(rawpath, 'drce_finite' + theta_w_str + 'and' + theta_v_str + '.pkl'), output_drce_list)
                save_data(os.path.join(rawpath, 'drce_infinite' + theta_w_str + 'and' + theta_v_str + '.pkl'), output_inf_drce_list)
            save_data(os.path.join(rawpath, 'lqg.pkl'), output_lqg_list)
    
    # Create combinations over the parameter list
    combinations = [(dist_parameter, theta, idx_w, idx_v)
                    for idx_w, dist_parameter in enumerate(dist_parameter_list)
                    for idx_v, theta in enumerate(theta_v_list)]
    
    results = Parallel(n_jobs=-1)(
        delayed(perform_simulation)(lambda_, noise_dist, dist_parameter, theta, idx_w, idx_v)
        for dist_parameter, theta, idx_w, idx_v in combinations
    )
    
    if not use_lambda:
        path = "./results/{}_{}/experiment/params_thetas/".format(dist, noise_dist)
        save_data(path + 'nonzero_drce_lambda.pkl', DRCE_lambda)
    
    print("Params data generation Completed!")
    if use_lambda:
        print("Now run: python plot_params_EM_infDRCE.py --use_lambda --dist " + dist + " --noise_dist " + noise_dist)
    else:
        print("Now run: python plot_params_EM_infDRCE.py --dist " + dist + " --noise_dist " + noise_dist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str,help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', required=False, default="normal", type=str,help="Noise distribution (normal or quadratic)")
    parser.add_argument('--num_sim', required=False, default=500, type=int,help="Number of simulation runs")
    parser.add_argument('--num_samples', required=False, default=15, type=int,help="Number of disturbance samples (not used in this experiment)")
    parser.add_argument('--num_noise_samples', required=False, default=15, type=int,help="Number of noise samples (not used in this experiment)")
    parser.add_argument('--horizon', required=False, default=20, type=int,help="Horizon length")
    parser.add_argument('--infinite', required=False, action="store_true",help="Infinite horizon flag (not used in this experiment)")
    
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon, args.infinite)
