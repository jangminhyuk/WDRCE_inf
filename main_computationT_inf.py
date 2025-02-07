#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file measures the computation time (in seconds) for the combined 
controller instantiation and backward() method for the DRCE and inf_DRCE 
controllers as a function of the simulation horizon T.
We use lambda = 10.0 and use_lambda=True.
For each time horizon T (e.g., 5, 10, …, 50), the code:
  1. Generates data and runs EM once to obtain nominal parameters.
  2. Prepares the nominal parameters.
  3. For a fixed number of repetitions (num_reps), re-instantiates each controller
     (DRCE and inf_DRCE) and measures the time from before instantiation to after 
     backward() returns.
The average and standard deviation of these times are saved.
Results are stored in "./results/backward_time_experiment_lambda/".
"""

import numpy as np
import argparse
from controllers.inf_DRCE import inf_DRCE
from controllers.DRCE import DRCE
from numpy.linalg import norm
from pykalman import KalmanFilter
import os
import pickle
import time

reg_eps = 1e-6

# --- Utility Functions (from your reference code) ---
def uniform(a, b, N=1):
    n = a.shape[0]
    x = a + (b - a)*np.random.rand(N, n)
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
                x[i][j] = beta + tmp**(1./3.)
            else:
                x[i][j] = beta - (-tmp)**(1./3.)
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
    mean_ = np.average(w, axis=1)
    diff = (w.T - mean_)[...,np.newaxis]
    var_ = np.average((diff @ np.transpose(diff, (0,2,1))), axis=0)
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
    mean_ = np.average(w, axis=1)[...,np.newaxis]
    var_ = np.cov(w)
    return mean_, var_

def save_data(path, data):
    with open(path, 'wb') as output:
        pickle.dump(data, output)

def generate_data(T, nx, ny, nu, A, B, C, mu_w, Sigma_w, mu_v, M,
                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist):
    u = np.zeros((T, nu, 1))
    x_true_all = np.zeros((T + 1, nx, 1))
    y_all = np.zeros((T, ny, 1))
    if dist=="normal":
        x_true = normal(x0_mean, x0_cov)
    elif dist=="quadratic":
        x_true = quadratic(x0_max, x0_min)
    x_true_all[0] = x_true
    for t in range(T):
        if dist=="normal":
            true_w = normal(mu_w, Sigma_w)
            true_v = normal(mu_v, M)
        elif dist=="quadratic":
            true_w = quadratic(w_max, w_min)
            true_v = quadratic(v_max, v_min)
        y_t = C @ x_true + true_v
        y_all[t] = y_t
        x_true = A @ x_true + B @ u[t] + true_w
        x_true_all[t + 1] = x_true
    return x_true_all, y_all

# --- Backward Time Measurement Experiment ---
# For each horizon T, we generate data and run EM once.
# Then, for num_reps repetitions, we measure the time (instantiation + backward) for:
#    DRCE and inf_DRCE, with lambda=10.0 and use_lambda=True.
def measure_backward_time_for_horizon(T, num_reps, dist, noise_dist, use_lambda=True):
    fixed_theta_w = 1.0
    fixed_theta_v = 1.0
    # --- System Initialization ---
    nx = 10; nu = 10; ny = 10
    temp = np.ones((nx, nx))
    A = 0.2 * (np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2))
    B = np.eye(nx)
    C = Q = R = Qf = np.eye(nx)
    # --- Disturbance Distribution ---
    if dist=="normal":
        w_max = None; w_min = None
        mu_w = -0.5 * np.ones((nx,1))
        Sigma_w = 0.5 * np.eye(nx)
        x0_max = None; x0_min = None
        x0_mean = 0.2 * np.ones((nx,1))
        x0_cov = 0.1 * np.eye(nx)
    elif dist=="quadratic":
        w_max = 1.0 * np.ones(nx)
        w_min = -2.0 * np.ones(nx)
        mu_w = (0.5*(w_max+w_min))[..., np.newaxis]
        Sigma_w = 3.0/20.0 * np.diag((w_max-w_min)**2)
        x0_max = 0.21 * np.ones(nx)
        x0_min = 0.19 * np.ones(nx)
        x0_mean = (0.5*(x0_max+x0_min))[..., np.newaxis]
        x0_cov = 3.0/20.0 * np.diag((x0_max-x0_min)**2)
    # --- Noise Distribution ---
    if noise_dist=="normal":
        v_max = None; v_min = None
        M = 1.0 * np.eye(ny)
        mu_v = 0.5 * np.ones((ny,1))
    elif noise_dist=="quadratic":
        v_min = -1.0 * np.ones(ny)
        v_max = 2.0 * np.ones(ny)
        mu_v = (0.5*(v_max+v_min))[..., np.newaxis]
        M = 3.0/20.0 * np.diag((v_max-v_min)**2)
    # --- Generate Data and run EM once ---
    N = 50
    x_all, y_all = generate_data(N, nx, ny, nu, A, B, C,
                                  mu_w, Sigma_w, mu_v, M,
                                  x0_mean, x0_cov, x0_max, x0_min,
                                  w_max, w_min, v_max, v_min, dist)
    y_all = y_all.squeeze()
    # Run EM to estimate nominal parameters
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
    max_iter = 50; eps_log = 1e-4; prev_ll = -np.inf
    for i in range(max_iter):
        kf = kf.em(X=y_all, n_iter=1)
        ll = kf.loglikelihood(y_all)
        if i > 0 and (ll - prev_ll <= eps_log):
            break
        prev_ll = ll
    Sigma_w_hat = kf.transition_covariance
    Sigma_v_hat = kf.observation_covariance
    mu_w_hat = np.array(kf.transition_offsets).reshape(-1,1)
    mu_v_hat = np.array(kf.observation_offsets).reshape(-1,1)
    M_hat = Sigma_v_hat
    # --- Prepare nominal parameters for forward simulation ---
    mu_w_hat_fh = np.tile(mu_w_hat, (T, 1, 1))
    mu_v_hat_fh = np.tile(mu_v_hat, (T+1, 1, 1))
    Sigma_w_hat_fh = np.tile(Sigma_w_hat, (T, 1, 1))
    M_hat_fh = np.tile(M_hat, (T+1, 1, 1))
    x0_mean_hat = x0_mean
    x0_cov_hat = x0_cov
    # --- Controller Parameters ---
    lambda_ = 10.0
    theta_x0 = 0.5
    system_data = (A, B, C, Q, Qf, R, M)
    # --- Measure time including instantiation and backward() ---
    drce_times = []
    inf_drce_times = []
    for rep in range(num_reps):
        # DRCE 
        start = time.perf_counter()
        drce_rep = DRCE(lambda_, fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                        system_data, mu_w_hat_fh, Sigma_w_hat_fh,
                        x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                        v_max, v_min, mu_v, mu_v_hat_fh, M_hat_fh, x0_mean_hat, x0_cov_hat,
                        True, False)
        drce_rep.backward()
        end = time.perf_counter()
        drce_times.append(end - start)
        
        # inf_DRCE
        start = time.perf_counter()
        inf_drce_rep = inf_DRCE(lambda_, fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                                system_data, mu_w_hat, Sigma_w_hat,
                                x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                                v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat,
                                True, False)
        inf_drce_rep.backward()
        end = time.perf_counter()
        inf_drce_times.append(end - start)
        
    avg_drce_time = np.mean(drce_times)
    std_drce_time = np.std(drce_times)
    avg_inf_drce_time = np.mean(inf_drce_times)
    std_inf_drce_time = np.std(inf_drce_times)
    
    return avg_drce_time, std_drce_time, avg_inf_drce_time, std_inf_drce_time

def main(dist, noise_dist, num_sim, num_reps):
    horizon_list = list(range(5, 51, 5))
    summary = {"T": [],
               "DRCE": {"mean": [], "std": []},
               "inf_DRCE": {"mean": [], "std": []}}
    for T in horizon_list:
        print(f"Computation time for T = {T}")
        avg_drce_time, std_drce_time, avg_inf_drce_time, std_inf_drce_time = measure_backward_time_for_horizon(T, num_reps, dist, noise_dist, use_lambda=True)
        summary["T"].append(T)
        summary["DRCE"]["mean"].append(avg_drce_time)
        summary["DRCE"]["std"].append(std_drce_time)
        summary["inf_DRCE"]["mean"].append(avg_inf_drce_time)
        summary["inf_DRCE"]["std"].append(std_inf_drce_time)
    save_folder = "./results/backward_time_experiment_lambda/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_data(os.path.join(save_folder, f"backward_time_costs_{dist}_{noise_dist}.pkl"), summary)
    print("Backward time experiment data generation completed!")
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str,
                        help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', required=False, default="normal", type=str,
                        help="Noise distribution (normal or quadratic)")
    parser.add_argument('--num_sim', required=False, default=100, type=int,
                        help="(Not used) Number of forward simulation runs; not used here")
    parser.add_argument('--num_reps', required=False, default=10, type=int,
                        help="Number of repetitions for backward() measurement")
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_reps)
