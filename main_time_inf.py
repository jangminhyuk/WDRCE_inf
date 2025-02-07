#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file generates data for a time-horizon experiment comparing the control cost
(as computed by LQG, finite–horizon DRCE, and infinite–horizon DRCE) as a function of
the simulation horizon T. The ambiguity parameters are fixed:
    theta_w = 1.0 and theta_v = 1.0.
When the flag --use_lambda is provided, the controllers are instantiated with
use_lambda=True and the value λ = 10.0 is used.
For each horizon T (e.g., 5, 10, …, 50):
  1. Data is generated and nominal parameters are estimated once (via EM).
  2. The controllers are instantiated (with backward() called) using these nominal parameters.
  3. Then, num_sim forward simulations are run (using the same nominal distribution).
The summary (average and std of cost) is saved in a folder that depends on the mode.
"""

import numpy as np
import argparse
from controllers.inf_DRCE import inf_DRCE
from controllers.LQG import LQG
from controllers.DRCE import DRCE
from numpy.linalg import norm
from pykalman import KalmanFilter
import os
import pickle

reg_eps = 1e-6

# === Utility Functions (same as in your reference code) ===
def uniform(a, b, N=1):
    n = a.shape[0]
    x = a + (b-a)*np.random.rand(N, n)
    return x.T

def normal(mu, Sigma, N=1):
    x = np.random.multivariate_normal(mu[:,0], Sigma, size=N).T
    return x

def quad_inverse(x, b, a):
    row = x.shape[0]
    col = x.shape[1]
    for i in range(row):
        for j in range(col):
            beta = (a[j]+b[j]) / 2.0
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
    diff = (w.T - mean_)[..., np.newaxis]
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

# --- Time-Horizon Experiment Function ---
# For each time horizon T, we:
# 1. Generate data and run EM to obtain nominal parameters
# 2. Using these nominal parameters, run num_sim forward simulations.
def run_experiment_for_horizon(T, num_sim, dist, noise_dist, use_lambda):
    fixed_theta_w = 1.0
    fixed_theta_v = 1.0
    # --- System Initialization ---
    # nx = 10; nu = 10; ny = 10
    # temp = np.ones((nx, nx))
    # A = 0.2 * (np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2))
    # B = np.eye(nx)
    # C = Q = R = Qf = np.eye(nx)
    nx = 10 #state dimension
    nu = 10 #control input dimension
    ny = 9#output dimension
    temp = np.ones((nx, nx))
    A = np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2)
    B = Q = R = Qf = np.eye(10)
    C = np.hstack([np.eye(9), np.zeros((9,1))])
    # --- Disturbance Distribution ---
    if dist=="normal":
        w_max = None; w_min = None
        mu_w = 0.0 * np.ones((nx,1))
        Sigma_w = 0.1 * np.eye(nx)
        x0_max = None; x0_min = None
        x0_mean = 0.1 * np.ones((nx,1))
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
        M = 0.2 * np.eye(ny)
        mu_v = 0.0 * np.ones((ny,1))
    elif noise_dist=="quadratic":
        v_min = -1.0 * np.ones(ny)
        v_max = 2.0 * np.ones(ny)
        mu_v = (0.5*(v_max+v_min))[..., np.newaxis]
        M = 3.0/20.0 * np.diag((v_max-v_min)**2)
    # --- Generate Data ---
    N = 50
    x_all, y_all = generate_data(N, nx, ny, nu, A, B, C,
                                  mu_w, Sigma_w, mu_v, M,
                                  x0_mean, x0_cov, x0_max, x0_min,
                                  w_max, w_min, v_max, v_min, dist)
    y_all = y_all.squeeze()
    # --- Run EM once to obtain nominal parameters ---
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
        if i>0 and (ll - prev_ll <= eps_log):
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
    lambda_ = 20.0
    theta_x0 = 0.5
    system_data = (A, B, C, Q, Qf, R, M)
    # Instantiate controllers (using T exactly as in the reference code)
    lqg = LQG(T, dist, noise_dist, system_data, mu_w_hat_fh, Sigma_w_hat_fh,
              x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
              v_max, v_min, mu_v, mu_v_hat_fh, M_hat_fh, x0_mean_hat, x0_cov_hat)
    if use_lambda:
        drce = DRCE(lambda_, fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                    system_data, mu_w_hat_fh, Sigma_w_hat_fh,
                    x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                    v_max, v_min, mu_v, mu_v_hat_fh, M_hat_fh, x0_mean_hat, x0_cov_hat,
                    True, False)
        inf_drce = inf_DRCE(lambda_, fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                             system_data, mu_w_hat, Sigma_w_hat,
                             x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                             v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat,
                             True, False)
    else:
        drce = DRCE(lambda_, fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                    system_data, mu_w_hat_fh, Sigma_w_hat_fh,
                    x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                    v_max, v_min, mu_v, mu_v_hat_fh, M_hat_fh, x0_mean_hat, x0_cov_hat,
                    False, False)
        inf_drce = inf_DRCE(lambda_, fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                             system_data, mu_w_hat, Sigma_w_hat,
                             x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                             v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat,
                             False, False)
    lqg.backward()
    drce.backward()
    inf_drce.backward()
    
    # --- Run forward simulations using the same nominal distribution ---
    lqg_costs = []
    drce_costs = []
    inf_drce_costs = []
    for i in range(num_sim):
        out_lqg = lqg.forward()
        out_drce = drce.forward()
        out_inf_drce = inf_drce.forward()
        lqg_costs.append(out_lqg['cost'][0])
        drce_costs.append(out_drce['cost'][0])
        inf_drce_costs.append(out_inf_drce['cost'][0])
    avg_lqg = np.mean(lqg_costs)
    avg_drce = np.mean(drce_costs)
    avg_inf_drce = np.mean(inf_drce_costs)
    std_lqg = np.std(lqg_costs)
    std_drce = np.std(drce_costs)
    std_inf_drce = np.std(inf_drce_costs)
    return avg_lqg, std_lqg, avg_drce, std_drce, avg_inf_drce, std_inf_drce

def main(dist, noise_dist, num_sim, use_lambda_flag):
    horizon_list = list(range(2, 51, 2))
    summary = {"T": [], 
               "LQG": {"mean": [], "std": []},
               "DRCE_finite": {"mean": [], "std": []},
               "DRCE_infinite": {"mean": [], "std": []}}
    for T in horizon_list:
        print(f"Running experiment for time horizon T = {T}")
        avg_lqg, std_lqg, avg_drce, std_drce, avg_inf_drce, std_inf_drce = run_experiment_for_horizon(T, num_sim, dist, noise_dist, use_lambda_flag)
        summary["T"].append(T)
        summary["LQG"]["mean"].append(avg_lqg)
        summary["LQG"]["std"].append(std_lqg)
        summary["DRCE_finite"]["mean"].append(avg_drce)
        summary["DRCE_finite"]["std"].append(std_drce)
        summary["DRCE_infinite"]["mean"].append(avg_inf_drce)
        summary["DRCE_infinite"]["std"].append(std_inf_drce)
    if use_lambda_flag:
        save_folder = "./results/time_horizon_experiment_lambda/"
    else:
        save_folder = "./results/time_horizon_experiment/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_data(os.path.join(save_folder, f"time_horizon_costs_{dist}_{noise_dist}.pkl"), summary)
    print("Time horizon experiment data generation completed!")
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str,
                        help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', required=False, default="normal", type=str,
                        help="Noise distribution (normal or quadratic)")
    parser.add_argument('--num_sim', required=False, default=100, type=int,
                        help="Number of forward simulation runs for each horizon")
    parser.add_argument('--use_lambda', required=False, action="store_true",
                        help="Directly use lambda")
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.use_lambda)
