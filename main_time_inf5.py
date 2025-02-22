#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file generates data for a time-horizon experiment comparing the control cost
(as computed by LQG, inf_LQG, finite–horizon DRCE, infinite–horizon DRCE, and finite–horizon DRLQC)
as a function of the simulation horizon T. The ambiguity parameters are fixed:
    theta_w = 1.0 and theta_v = 1.0.
When the flag --use_lambda is provided, the controllers are instantiated with
use_lambda=True and the value λ = 20.0 is used.
For each horizon T (e.g., 2, 4, …, 50):
  1. Data is generated and nominal parameters are estimated once (via EM).
  2. Using these nominal parameters, instantiate five controllers:
       - Finite–horizon LQG (using tiled parameters)
       - Infinite–horizon LQG (using non–tiled parameters)
       - Finite–horizon DRCE (using tiled parameters)
       - Infinite–horizon DRCE (using non–tiled parameters)
       - Finite–horizon DRLQC (using tiled parameters)
  3. Then, num_sim forward simulations are run (using the same nominal distribution).
Additionally, the entire experiment is repeated N=10 times for each time horizon.
The summary (average and std of cost) is saved in a folder whose name includes "experiment_5".
"""

import numpy as np
import argparse
from controllers.inf_DRCE import inf_DRCE
from controllers.inf_LQG import inf_LQG
from controllers.DRCE import DRCE
from controllers.LQG import LQG
# --- DRLQC Import (added) ---
from controllers.DRLQC import DRLQC
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
            beta = (a[j] + b[j]) / 2.0
            alpha = 12.0/((b[j]-a[j])**3)
            tmp = 3*x[i][j]/alpha - (beta - a[j])**3
            if 0 <= tmp:
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
    var_ = np.average((diff @ np.transpose(diff, (0,2,1))), axis=0)
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

def save_data(path, data):
    with open(path, 'wb') as output:
        pickle.dump(data, output)

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

# --- Time-Horizon Experiment Function ---
# For each time horizon T, we:
# 1. Generate data and run EM to obtain nominal parameters.
# 2. Using these nominal parameters, instantiate five controllers:
#    - Finite–horizon LQG (using tiled parameters)
#    - Infinite–horizon LQG (using non-tiled parameters)
#    - Finite–horizon DRCE (using tiled parameters)
#    - Infinite–horizon DRCE (using non-tiled parameters)
#    - Finite–horizon DRLQC (using tiled parameters)  <-- (added)
# 3. Run num_sim forward simulations and record the cost.
def run_experiment_for_horizon(T, num_sim, dist, noise_dist, use_lambda):
    
    # --- System Initialization ---
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
    
    # --- Disturbance Distribution ---
    if dist == "normal":
        w_max = None; w_min = None
        mu_w = 0.02 * np.ones((nx, 1))
        Sigma_w = 0.02 * np.eye(nx)
        x0_max = None; x0_min = None
        x0_mean = 0.01 * np.ones((nx, 1))
        x0_cov = 0.01 * np.eye(nx)
    elif dist == "quadratic":
        w_max = 1.0 * np.ones(nx)
        w_min = -2.0 * np.ones(nx)
        mu_w = (0.5*(w_max+w_min))[..., np.newaxis]
        Sigma_w = 3.0/20.0 * np.diag((w_max-w_min)**2)
        x0_max = 0.21 * np.ones(nx)
        x0_min = 0.19 * np.ones(nx)
        x0_mean = (0.5*(x0_max+x0_min))[..., np.newaxis]
        x0_cov = 3.0/20.0 * np.diag((x0_max-x0_min)**2)
    # --- Noise Distribution ---
    if noise_dist == "normal":
        v_max = None; v_min = None
        M = 0.01 * np.eye(ny)
        mu_v = 0.01 * np.ones((ny, 1))
    elif noise_dist == "quadratic":
        v_min = -1.0 * np.ones(ny)
        v_max = 2.0 * np.ones(ny)
        mu_v = (0.5*(v_max+v_min))[..., np.newaxis]
        M = 3.0/20.0 * np.diag((v_max-v_min)**2)
    # --- Controller Parameters ---
    fixed_theta_w = 0.5
    fixed_theta_v = 0.5
    lambda_ = 30.0
    theta_x0 = 0.01
    
    # --- Generate Data ---
    N = 5
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
        if i > 0 and (ll - prev_ll <= eps_log):
            break
        prev_ll = ll
    Sigma_w_hat = kf.transition_covariance
    Sigma_v_hat = kf.observation_covariance
    mu_w_hat = np.array(kf.transition_offsets).reshape(-1, 1)
    mu_v_hat = np.array(kf.observation_offsets).reshape(-1, 1)
    M_hat = Sigma_v_hat
    # --- Prepare nominal parameters for forward simulation ---
    # For finite-horizon controllers we tile the parameters over T.
    mu_w_hat_fh = np.tile(mu_w_hat, (T, 1, 1))
    mu_v_hat_fh = np.tile(mu_v_hat, (T+1, 1, 1))
    Sigma_w_hat_fh = np.tile(Sigma_w_hat, (T, 1, 1))
    M_hat_fh = np.tile(M_hat, (T+1, 1, 1))
    x0_mean_hat = x0_mean
    x0_cov_hat = x0_cov
    
    system_data = (A, B, C, Q, Qf, R, M)
    # --- Instantiate Controllers ---
    # Finite-horizon LQG (using tiled parameters)
    lqg_fh = LQG(T, dist, noise_dist, system_data, mu_w_hat_fh, Sigma_w_hat_fh,
                 x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                 v_max, v_min, mu_v, mu_v_hat_fh, M_hat_fh, x0_mean_hat, x0_cov_hat)
    # Infinite-horizon LQG (using non-tiled parameters)
    lqg_inf = inf_LQG(T, dist, noise_dist, system_data, mu_w_hat, Sigma_w_hat,
                      x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                      v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat)
    # Finite-horizon DRCE (using tiled parameters)
    if use_lambda:
        drce_fh = DRCE(lambda_, fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                       system_data, mu_w_hat_fh, Sigma_w_hat_fh,
                       x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                       v_max, v_min, mu_v, mu_v_hat_fh, M_hat_fh, x0_mean_hat, x0_cov_hat,
                       True, False)
        # Infinite-horizon DRCE (using non-tiled parameters)
        drce_inf = inf_DRCE(lambda_, fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                            system_data, mu_w_hat, Sigma_w_hat,
                            x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                            v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat,
                            True, False)
    else:
        drce_fh = DRCE(lambda_, fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                       system_data, mu_w_hat_fh, Sigma_w_hat_fh,
                       x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                       v_max, v_min, mu_v, mu_v_hat_fh, M_hat_fh, x0_mean_hat, x0_cov_hat,
                       False, False)
        drce_inf = inf_DRCE(lambda_, fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                            system_data, mu_w_hat, Sigma_w_hat,
                            x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min,
                            v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat,
                            False, False)
                            
    # --- Instantiate Finite-horizon DRLQC (using tiled parameters) ---  # (DRLQC added)
    tol = 1e-2
    # Construct batch matrices for DRLQC:
    W_hat = np.zeros((nx, nx, T+1))
    V_hat = np.zeros((ny, ny, T+1))
    for i in range(T):
        W_hat[:,:,i] = Sigma_w_hat
    for i in range(T+1):
        V_hat[:,:,i] = M_hat
    drlqc = DRLQC(fixed_theta_w, fixed_theta_v, theta_x0, T, dist, noise_dist,
                  system_data, mu_w_hat_fh, W_hat, x0_mean, x0_cov, x0_max, x0_min,
                  mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, mu_v_hat_fh,
                  V_hat, x0_mean_hat, x0_cov_hat, tol)
    
    # --- Compute the control policy (backward recursion) ---
    lqg_fh.backward()
    lqg_inf.backward()
    drce_fh.backward()
    drce_inf.backward()
    # For DRLQC, first solve the SDP and then run backward pass  (DRLQC added)
    drlqc.solve_sdp()
    drlqc.backward()
    
    # --- Run forward simulations using the same nominal distribution ---
    lqg_fh_costs = []
    lqg_inf_costs = []
    drce_fh_costs = []
    drce_inf_costs = []
    drlqc_costs = []  # (DRLQC added)
    for i in range(num_sim):
        out_lqg_fh = lqg_fh.forward()
        out_lqg_inf = lqg_inf.forward()
        out_drce_fh = drce_fh.forward()
        out_drce_inf = drce_inf.forward()
        out_drlqc = drlqc.forward()  # (DRLQC added)
        lqg_fh_costs.append(out_lqg_fh['cost'][0])
        lqg_inf_costs.append(out_lqg_inf['cost'][0])
        drce_fh_costs.append(out_drce_fh['cost'][0])
        drce_inf_costs.append(out_drce_inf['cost'][0])
        drlqc_costs.append(out_drlqc['cost'][0])  # (DRLQC added)
    avg_lqg_fh = np.mean(lqg_fh_costs)
    std_lqg_fh = np.std(lqg_fh_costs)
    avg_lqg_inf = np.mean(lqg_inf_costs)
    std_lqg_inf = np.std(lqg_inf_costs)
    avg_drce_fh = np.mean(drce_fh_costs)
    std_drce_fh = np.std(drce_fh_costs)
    avg_drce_inf = np.mean(drce_inf_costs)
    std_drce_inf = np.std(drce_inf_costs)
    avg_drlqc = np.mean(drlqc_costs)  
    std_drlqc = np.std(drlqc_costs)  
    return (avg_lqg_fh, std_lqg_fh, avg_lqg_inf, std_lqg_inf,
            avg_drce_fh, std_drce_fh, avg_drce_inf, std_drce_inf,
            avg_drlqc, std_drlqc)  # (DRLQC added)

def main(dist, noise_dist, num_sim, use_lambda_flag):
    horizon_list = list(range(12, 31, 2))
    num_experiments = 10  # Repeat the entire experiment 10 times for each T
    summary = {"T": [],
               "LQG_finite": {"mean": [], "std": []},
               "LQG_infinite": {"mean": [], "std": []},
               "DRCE_finite": {"mean": [], "std": []},
               "DRCE_infinite": {"mean": [], "std": []},
               "DRLQC_finite": {"mean": [], "std": []}}
    
    for T in horizon_list:
        print(f"Running experiment for time horizon T = {T}")
        # Lists to store the average cost from each replication for each controller.
        lqg_fh_rep = []
        lqg_inf_rep = []
        drce_fh_rep = []
        drce_inf_rep = []
        drlqc_rep = []  
        # Repeat the entire experiment num_experiments times.
        for rep in range(num_experiments):
            # Fix the random seed for reproducibility.
            np.random.seed(2024 + rep)
            (avg_lqg_fh, std_lqg_fh, avg_lqg_inf, std_lqg_inf,
             avg_drce_fh, std_drce_fh, avg_drce_inf, std_drce_inf,
             avg_drlqc, std_drlqc) = run_experiment_for_horizon(T, num_sim, dist, noise_dist, use_lambda_flag)
            lqg_fh_rep.append(avg_lqg_fh)
            lqg_inf_rep.append(avg_lqg_inf)
            drce_fh_rep.append(avg_drce_fh)
            drce_inf_rep.append(avg_drce_inf)
            drlqc_rep.append(avg_drlqc)  # (DRLQC added)
        # Aggregate the results over the replications.
        final_avg_lqg_fh = np.mean(lqg_fh_rep)
        final_std_lqg_fh = np.std(lqg_fh_rep)
        final_avg_lqg_inf = np.mean(lqg_inf_rep)
        final_std_lqg_inf = np.std(lqg_inf_rep)
        final_avg_drce_fh = np.mean(drce_fh_rep)
        final_std_drce_fh = np.std(drce_fh_rep)
        final_avg_drce_inf = np.mean(drce_inf_rep)
        final_std_drce_inf = np.std(drce_inf_rep)
        final_avg_drlqc = np.mean(drlqc_rep)   # (DRLQC added)
        final_std_drlqc = np.std(drlqc_rep)     # (DRLQC added)
        
        summary["T"].append(T)
        summary["LQG_finite"]["mean"].append(final_avg_lqg_fh)
        summary["LQG_finite"]["std"].append(final_std_lqg_fh)
        summary["LQG_infinite"]["mean"].append(final_avg_lqg_inf)
        summary["LQG_infinite"]["std"].append(final_std_lqg_inf)
        summary["DRCE_finite"]["mean"].append(final_avg_drce_fh)
        summary["DRCE_finite"]["std"].append(final_std_drce_fh)
        summary["DRCE_infinite"]["mean"].append(final_avg_drce_inf)
        summary["DRCE_infinite"]["std"].append(final_std_drce_inf)
        summary["DRLQC_finite"]["mean"].append(final_avg_drlqc)   # (DRLQC added)
        summary["DRLQC_finite"]["std"].append(final_std_drlqc)     # (DRLQC added)
    
    if use_lambda_flag:
        save_folder = "./results/time_horizon_experiment_5_lambda/"  # (folder updated to 5)
    else:
        save_folder = "./results/time_horizon_experiment_5/"           # (folder updated to 5)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_data(os.path.join(save_folder, f"time_horizon_costs_exp5_{dist}_{noise_dist}.pkl"), summary)
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
