#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import argparse
from controllers.inf_DRCE import inf_DRCE
from controllers.inf_LQG import inf_LQG
from controllers.inf_WDRC import inf_WDRC
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
                x[i][j] = beta + tmp ** (1. / 3.)
            else:
                x[i][j] = beta - (-tmp) ** (1. / 3.)
    return x

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

# Function to generate the true states and measurements
def generate_data(T, nx, ny, nu, A, B, C, mu_w, Sigma_w, mu_v, M,
                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist):
    u = np.zeros((T, nu, 1))
    x_true_all = np.zeros((T + 1, nx, 1))
    y_all = np.zeros((T, ny, 1))
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
    num_experiments = 10  # Repeat nominal distribution generation 10 times

    # Global lists for final averaged simulation results.
    output_J_inf_LQG_mean = []      # For inf_LQG
    output_J_inf_LQG_std = []
    output_J_inf_WDRC_mean = []     # For inf_WDRC
    output_J_inf_WDRC_std = []
    output_J_inf_DRCE_mean = []     # For inf_DRCE
    output_J_inf_DRCE_std = []
    
    # ------- System Initialization -------
    # nx = 10  # state dimension
    # nu = 10  # control input dimension
    # ny = 9   # output dimension
    # temp = np.ones((nx, nx))
    # A = np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2)
    # B = Q = R = Qf = np.eye(10)
    # C = np.hstack([np.eye(9), np.zeros((9, 1))])

    # nx = 2
    # nu = 1
    # ny = 1
    # dt = 0.1
    # A = np.array([[1, dt],
    #               [0, 1]])
    # B = np.array([[0],
    #               [dt]])
    # C = np.array([[1,0]])
    # Q = Qf = np.eye(nx)
    # R = np.eye(nu)

    # Example 1: AC1 (COMPlib companion–form system)
    nx = 5
    nu = 3
    ny = 3

    A = np.array([
        [0,       0,      1.132,     0,      -1],
        [0,  -0.0538,   -0.1712,     0,   0.0705],
        [0,       0,         0,     1,       0],
        [0,   0.0485,         0, -0.8556, -1.013],
        [0,  -0.2909,         0,  1.0532, -0.6859]
    ])

    B = np.array([
        [0,      0,      0],
        [-0.12,  1,      0],
        [0,      0,      0],
        [4.419,  0, -1.665],
        [1.575,  0, -0.0732]
    ])

    C = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0]
    ])

    Q  = Qf = np.eye(nx)
    R  = np.eye(nu)

    # ----------------------------
    if dist == 'normal':
        theta_v_list = [0.1, 0.2, 0.5]
        theta_w_list = [0.1]
    else:
        theta_v_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        theta_w_list = [0.1, 0.2, 0.5, 1.0, 1.5]
    # lambda_list = [12, 15, 20, 25, 30, 35, 40]
    lambda_list = [20, 30, 40, 50, 60]
    theta_w_local = 1.0
    theta_x0 = 0.01
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
    
    # ------- Disturbance Distribution -------
    if dist == "normal":
        w_max = None
        w_min = None
        mu_w = 0.01 * np.ones((nx, 1))
        Sigma_w = 0.01 * np.eye(nx)
        x0_max = None
        x0_min = None
        x0_mean = 0.01 * np.ones((nx, 1))
        x0_cov = 0.01 * np.eye(nx)
    elif dist == "quadratic":
        w_max = 0.3 * np.ones(nx)
        w_min = -0.6 * np.ones(nx)
        mu_w = (0.5 * (w_max + w_min))[..., np.newaxis]
        Sigma_w = 3.0 / 20.0 * np.diag((w_max - w_min) ** 2)
        x0_max = 0.11 * np.ones(nx)
        x0_min = 0.09 * np.ones(nx)
        x0_mean = (0.5 * (x0_max + x0_min))[..., np.newaxis]
        x0_cov = 3.0 / 20.0 * np.diag((x0_max - x0_min) ** 2)
    # ------- Noise Distribution ---------
    if noise_dist == "normal":
        v_max = None
        v_min = None
        M = 0.02 * np.eye(ny)
        mu_v = 0.02 * np.ones((ny, 1))
    elif noise_dist == "quadratic":
        v_min = -0.5 * np.ones(ny)
        v_max = 1.0 * np.ones(ny)
        mu_v = (0.5 * (v_max + v_min))[..., np.newaxis]
        M = 3.0 / 20.0 * np.diag((v_max - v_min) ** 2)
    
    print(f'Real data:\n mu_w: {mu_w},\n mu_v: {mu_v},\n Sigma_w: {Sigma_w},\n Sigma_v: {M}')
    
    N = 10
    # (We generate data here only to check dimensions.)
    x_all, y_all = generate_data(N, nx, ny, nu, A, B, C, mu_w, Sigma_w, mu_v, M,
                                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist)
    y_all = y_all.squeeze()
    eps_param = 1e-5
    eps_log = 1e-4
    
    # Create a results directory (the same for all simulations)
    if use_lambda:
        path = "./results/{}_{}/experiment/params_lambda3/".format(dist, noise_dist)
    else:
        path = "./results/{}_{}/experiment/params_thetas3/".format(dist, noise_dist)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Modified simulation function for the three infinite-horizon controllers.
    def perform_simulation(lambda_, noise_dist, dist_parameter, theta, idx_w, idx_v):
        # For each parameter combination, repeat the entire experiment num_experiments times.
        rep_costs_inf_LQG = []
        rep_costs_inf_WDRC = []
        rep_costs_inf_DRCE = []
        for rep in range(num_experiments):
            # Set a new random seed for each repetition.
            np.random.seed(seed + rep)
            if use_lambda:
                lambda_local = dist_parameter
                theta_w_local = 1.0  # not used when use_lambda is True
            else:
                lambda_local = lambda_
                theta_w_local = dist_parameter
            
            # Generate new data and re-run EM to obtain nominal parameters.
            x_all, y_all = generate_data(N, nx, ny, nu, A, B, C,
                                          mu_w, Sigma_w, mu_v, M,
                                          x0_mean, x0_cov, x0_max, x0_min,
                                          w_max, w_min, v_max, v_min, dist)
            y_all = y_all.squeeze()
            mu_w_hat = np.zeros(nx)
            mu_v_hat = np.zeros(ny)
            mu_x0_hat = x0_mean.squeeze()
            Sigma_w_hat = np.eye(nx)
            Sigma_v_hat = np.eye(ny)
            Sigma_x0_hat = x0_cov
            kf = KalmanFilter(A, C, Sigma_w_hat, Sigma_v_hat, mu_w_hat, mu_v_hat,
                              mu_x0_hat, Sigma_x0_hat,
                              em_vars=['transition_covariance','observation_covariance',
                                       'transition_offsets','observation_offsets'])
            max_iter = 50
            loglikelihoods = np.zeros(max_iter)
            for i in range(max_iter):
                kf = kf.em(X=y_all, n_iter=1)
                loglikelihoods[i] = kf.loglikelihood(y_all)
                Sigma_w_hat = kf.transition_covariance
                Sigma_v_hat = kf.observation_covariance
                mu_w_hat = kf.transition_offsets
                mu_v_hat = kf.observation_offsets
                mu_x0_hat = kf.initial_state_mean
                Sigma_x0_hat = kf.initial_state_covariance
                if i > 0 and (loglikelihoods[i] - loglikelihoods[i-1] <= eps_log):
                    break
            mu_w_hat = np.array(mu_w_hat).reshape(-1, 1)
            mu_v_hat = np.array(mu_v_hat).reshape(-1, 1)
            M_hat = Sigma_v_hat

            # For infinite-horizon controllers, we use the steady-state estimates.
            x0_mean_hat = x0_mean
            x0_cov_hat = x0_cov
            
            # Create system data tuple.
            system_data = (A, B, C, Q, Qf, R, M)
            
            # Instantiate controllers.
            inf_lqg = inf_LQG(T, dist, noise_dist, system_data,
                              mu_w_hat, Sigma_w_hat,
                              x0_mean, x0_cov, x0_max, x0_min,
                              mu_w, Sigma_w, w_max, w_min,
                              v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat)
            
            inf_wdrc = inf_WDRC(lambda_local, theta_w_local, T, dist, noise_dist,
                                 system_data, mu_w_hat, Sigma_w_hat,
                                 x0_mean, x0_cov, x0_max, x0_min,
                                 mu_w, Sigma_w, w_max, w_min,
                                 v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat,
                                 use_lambda, use_optimal_lambda)
            
            inf_drce = inf_DRCE(lambda_local, theta_w_local, theta, theta_x0, T, dist, noise_dist,
                                 system_data, mu_w_hat, Sigma_w_hat,
                                 x0_mean, x0_cov, x0_max, x0_min,
                                 mu_w, Sigma_w, w_max, w_min,
                                 v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat,
                                 use_lambda, use_optimal_lambda)
            
            # Compute control policies.
            inf_lqg.backward()
            inf_wdrc.backward()
            inf_drce.backward()
            
            # Run forward simulations.
            out_inf_lqg_list = []
            out_inf_wdrc_list = []
            out_inf_drce_list = []
            print("Running inf_LQG Forward step ... (rep {})".format(rep))
            for i in range(num_sim):
                out_inf_lqg = inf_lqg.forward()
                out_inf_lqg_list.append(out_inf_lqg)
                if i % 50 == 0:
                    print("Simulation #", i, '| cost (inf_LQG):', out_inf_lqg['cost'][0])
            J_inf_LQG_list = [out['cost'] for out in out_inf_lqg_list]
            rep_costs_inf_LQG.append(np.mean(J_inf_LQG_list))
            
            print("Running inf_WDRC Forward step ... (rep {})".format(rep))
            for i in range(num_sim):
                out_inf_wdrc = inf_wdrc.forward()
                out_inf_wdrc_list.append(out_inf_wdrc)
                if i % 50 == 0:
                    print("Simulation #", i, '| cost (inf_WDRC):', out_inf_wdrc['cost'][0])
            J_inf_WDRC_list = [out['cost'] for out in out_inf_wdrc_list]
            rep_costs_inf_WDRC.append(np.mean(J_inf_WDRC_list))
            
            print("Running inf_DRCE Forward step ... (rep {})".format(rep))
            for i in range(num_sim):
                out_inf_drce = inf_drce.forward()
                out_inf_drce_list.append(out_inf_drce)
                if i % 50 == 0:
                    print("Simulation #", i, '| cost (inf_DRCE):', out_inf_drce['cost'][0])
            J_inf_DRCE_list = [out['cost'] for out in out_inf_drce_list]
            rep_costs_inf_DRCE.append(np.mean(J_inf_DRCE_list))
            
            # (Optional) Save raw simulation data for this repetition if desired.
            rawpath = os.path.join(path, "raw")
            if not os.path.exists(rawpath):
                os.makedirs(rawpath)
            if use_lambda:
                save_data(os.path.join(rawpath, 'inf_drce_' + str(lambda_local) + '.pkl'), out_inf_drce_list)
                save_data(os.path.join(rawpath, 'inf_wdrc_' + str(lambda_local) + '.pkl'), out_inf_wdrc_list)
            else:
                save_data(os.path.join(rawpath, 'inf_drce.pkl'), out_inf_drce_list)
                save_data(os.path.join(rawpath, 'inf_wdrc.pkl'), out_inf_wdrc_list)
            save_data(os.path.join(rawpath, 'inf_lqg.pkl'), out_inf_lqg_list)
        
        # After num_experiments repetitions, compute the mean and std for each controller.
        mean_inf_LQG = np.mean(rep_costs_inf_LQG)
        std_inf_LQG = np.std(rep_costs_inf_LQG)
        mean_inf_WDRC = np.mean(rep_costs_inf_WDRC)
        std_inf_WDRC = np.std(rep_costs_inf_WDRC)
        mean_inf_DRCE = np.mean(rep_costs_inf_DRCE)
        std_inf_DRCE = np.std(rep_costs_inf_DRCE)
        
        # Append these averaged values to the global lists.
        output_J_inf_LQG_mean.append(mean_inf_LQG)
        output_J_inf_LQG_std.append(std_inf_LQG)
        output_J_inf_WDRC_mean.append(mean_inf_WDRC)
        output_J_inf_WDRC_std.append(std_inf_WDRC)
        output_J_inf_DRCE_mean.append(mean_inf_DRCE)
        output_J_inf_DRCE_std.append(std_inf_DRCE)
        
        # Save averaged cost data for this parameter combination.
        theta_v_str = f"_{str(theta).replace('.', '_')}"
        theta_w_str = f"_{str(theta_w_local).replace('.', '_')}"
        if use_lambda:
            save_data(path + 'drce_finite_' + str(lambda_local) + 'and' + theta_v_str + '.pkl', mean_inf_DRCE)
            save_data(path + 'drce_infinite_' + str(lambda_local) + 'and' + theta_v_str + '.pkl', mean_inf_WDRC)
        else:
            save_data(path + 'drce_finite' + theta_w_str + 'and' + theta_v_str + '.pkl', mean_inf_DRCE)
            save_data(path + 'drce_infinite' + theta_w_str + 'and' + theta_v_str + '.pkl', mean_inf_WDRC)
        save_data(path + 'lqg.pkl', mean_inf_LQG)
    
    # Create combinations over the parameter list.
    combinations = [(dist_parameter, theta, idx_w, idx_v)
                    for idx_w, dist_parameter in enumerate(dist_parameter_list)
                    for idx_v, theta in enumerate(theta_v_list)]
    
    results = Parallel(n_jobs=-1)(
        delayed(perform_simulation)(lambda_, noise_dist, dist_parameter, theta, idx_w, idx_v)
        for dist_parameter, theta, idx_w, idx_v in combinations
    )
    
    if not use_lambda:
        path = "./results/{}_{}/experiment/params_thetas3/".format(dist, noise_dist)
        save_data(path + 'nonzero_drce_lambda.pkl', DRCE_lambda)
    
    print("Params data generation Completed!")
    if use_lambda:
        print("Now run: python plot_params_EM_infDRCE9_3.py --use_lambda --dist " + dist + " --noise_dist " + noise_dist)
    else:
        print("Now run: python plot_params_EM_infDRCE9_3.py --dist " + dist + " --noise_dist " + noise_dist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str,
                        help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', required=False, default="normal", type=str,
                        help="Noise distribution (normal or quadratic)")
    parser.add_argument('--num_sim', required=False, default=500, type=int,
                        help="Number of simulation runs")
    parser.add_argument('--num_samples', required=False, default=15, type=int,
                        help="Number of disturbance samples (not used in this experiment)")
    parser.add_argument('--num_noise_samples', required=False, default=15, type=int,
                        help="Number of noise samples (not used in this experiment)")
    parser.add_argument('--horizon', required=False, default=20, type=int,
                        help="Horizon length")
    parser.add_argument('--infinite', required=False, action="store_true",
                        help="Infinite horizon flag (not used in this experiment)")
    
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon, args.infinite)
