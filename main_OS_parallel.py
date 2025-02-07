#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.DRCE import DRCE
from joblib import Parallel, delayed

import os
import pickle

# Define the function to be parallelized
def perform_simulation(i, lambda_, num_sim, use_lambda, use_optimal_lambda, WDRC_lambda, DRCE_lambda, idx, theta_w, theta_v, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat):
    print("iteration ", i, "/", num_sim)
    
    if use_optimal_lambda:  # If we use pre-calculated lambda
        lambda_ = DRCE_lambda[i][0]
    drce_ = DRCE(lambda_, theta_w, theta_v, theta_x0, T, dist, noise_dist, system_data, mu_hat[i], Sigma_hat[i], x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat[i],  M_hat[i], x0_mean_hat[i][0], x0_cov_hat[i][0], use_lambda, use_optimal_lambda)
    drce_.backward()
    
    result = {
        'drce': drce_,
        'DRCE_lambda': drce_.lambda_ if not use_lambda else None
    }
    return result

def sample_forward(algo, i):
    return algo[i].forward()

def drce_forward_step(drce, i, os_sample_size):
    drce_cert_val = drce[i].objective(drce[i].lambda_) * drce[i].T
    output_drce_sample = Parallel(n_jobs=-1)(delayed(sample_forward)(drce,i) for _ in range(os_sample_size))
    return drce_cert_val, output_drce_sample

def wdrc_forward_step(wdrc, i, os_sample_size):
    output_wdrc_sample = Parallel(n_jobs=-1)(delayed(sample_forward)(wdrc,i) for _ in range(os_sample_size))
    return output_wdrc_sample

def lqg_forward_step(lqg, i, os_sample_size):
    output_lqg_sample = Parallel(n_jobs=-1)(delayed(sample_forward)(lqg, i) for _ in range(os_sample_size))
    return output_lqg_sample

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

    #w = np.random.choice([0, 1], size=(n,N))
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
    #noise_plot_results = True
    seed = 2024 # Random seed !  any value
    np.random.seed(seed) # fix Random seed!
    
    num_noise_list = [num_noise_samples]
    num_x0_samples = 15 # num x0 samples 
    
    output_J_DRCE_mean=[]
    output_J_DRCE_std=[]
    output_DRCE_prob = []
    #-------Initialization-------
    nx = 10 #state dimension
    nu = 10 #control input dimension
    ny = 10#output dimension
    temp = np.ones((nx, nx))
    A = np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2)
    B = C = Q = R = Qf = np.eye(10) 
    #----------------------------
    # change True to False if you don't want to use given lambda
    use_lambda = True
    use_optimal_lambda = False
    lambda_ = 20 # will not be used if the parameter "use_lambda = False"
    noisedist = [noise_dist1]
    
    
    theta_w_list = [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    num_noise_list = [10, 15, 20] 
    #theta=theta_w=theta_v=theta_x0 for Out-of-Sample Experiments
    for noise_dist in noisedist:
        for theta_w in theta_w_list:
                WDRC_lambda, DRCE_lambda = [],[]
                theta = theta_w
                theta_x0=theta
                #for theta in theta_v_list:
                for idx, num_noise in enumerate(num_noise_list):
                    # Use N = N_w = N_v = N_x0
                    num_samples = num_noise
                    num_x0_samples = num_noise
                    theta_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
                    
                    # Load Lambda
                    DRCE_lambda_file = open('./inputs/OutOfSample/N={}/drce_lambda_{}.pkl'.format(num_noise,theta_), 'rb')
                    DRCE_lambda = pickle.load(DRCE_lambda_file)
                    DRCE_lambda_file.close()
                    
                    
                    
                    print("disturbance : ", dist, "/ noise : ", noise_dist, "/ num_noise : ", num_noise, "/ theta_w : ", theta_w, "/ theta_v : ", theta)
                    np.random.seed(seed) # fix Random seed!
                    print("--------------------------------------------")
                    print("number of noise sample : ", num_noise)
                    print("number of disturbance sample : ", num_samples)
                    
                    path = "./results/{}_{}/finite/multiple/OS/N={}/".format(dist, noise_dist, num_noise)    
                    if not os.path.exists(path):
                        os.makedirs(path)
                
                    #-------Disturbance distribution-------
                    if dist == "normal":
                        #disturbance distribution parameters
                        w_max = None
                        w_min = None
                        mu_w = 0.1*np.ones((nx, 1))
                        Sigma_w= 0.1*np.eye(nx)
                        #initial state distribution parameters
                        x0_max = None
                        x0_min = None
                        x0_mean = 0.1*np.ones((nx,1))
                        x0_cov = 0.1*np.eye(nx)
                    elif dist == "quadratic":
                        #disturbance distribution parameters
                        w_max = 0.2*np.ones(nx)
                        w_min = -0.2*np.ones(nx)
                        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
                        Sigma_w = 3.0/20.0*np.diag((w_max - w_min)**2)
                        #initial state distribution parameters
                        x0_max = 0.2*np.ones(nx)
                        x0_min = 0.0*np.ones(nx)
                        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
                        x0_cov = 3.0/20.0 *np.diag((x0_max - x0_min)**2)
                        
                    #-------Noise distribution ---------#
                    if noise_dist =="normal":
                        v_max = None
                        v_min = None
                        M = 2.0*np.eye(ny) #observation noise covariance
                        mu_v = 0.5*np.ones((ny, 1))
                    elif noise_dist =="quadratic":
                        v_min = -1.0*np.ones(ny)
                        v_max = 1.0*np.ones(ny)
                        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                        M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance
                        
                        
                    #-------Estimate the nominal distribution-------
                    # Out-of-sample test
                    x0_mean_hat, x0_cov_hat = [], []
                    mu_hat, Sigma_hat = [], []
                    v_mean_hat, M_hat = [], []
                    for i in range(num_sim):
                        # Nominal initial state distribution
                        x0_mean_hat_, x0_cov_hat_ = gen_sample_dist(dist, 1, num_x0_samples, mu_w=x0_mean, Sigma_w=x0_cov, w_max=x0_max, w_min=x0_min)
                        # Nominal Disturbance distribution
                        mu_hat_, Sigma_hat_ = gen_sample_dist(dist, T+1, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
                        # Nominal Noise distribution
                        v_mean_hat_, M_hat_ = gen_sample_dist(noise_dist, T+1, num_noise, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min)
                        M_hat_ = M_hat_ + 1e-5*np.eye(ny) # to prevent numerical error from inverse in standard KF at small sample size
                        
                        x0_mean_hat.append(x0_mean_hat_)
                        x0_cov_hat.append(x0_cov_hat_)
                        mu_hat.append(mu_hat_)
                        Sigma_hat.append(Sigma_hat_)
                        v_mean_hat.append(v_mean_hat_)
                        M_hat.append(M_hat_)
                        
                    
                    #-------Create a random system-------
                    system_data = (A, B, C, Q, Qf, R, M)
                    
                    #-------Perform n  independent simulations and summarize the results-------
                    
                    
                    # Parallelize the simulations
                    results = Parallel(n_jobs=-1)(
                        delayed(perform_simulation)(i, lambda_, num_sim, use_lambda, use_optimal_lambda, WDRC_lambda, DRCE_lambda, idx, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat)
                        for i in range(num_sim)
                    )
                    
                        
                    # Process results
                    drce = []
                    
                    for result in results:
                        drce.append(result['drce'])
    
                    
                        
                    print('---------------------')
                    output_drce = np.empty(num_sim, dtype=object)
                    drce_cert = np.empty(num_sim)
                    os_sample_size = 1000
                    
                    
                    # Parallel execution for DRCE
                    print("Running DRCE Forward step ...")
                    results_drce = Parallel(n_jobs=-1)(delayed(drce_forward_step)(drce, i, os_sample_size) for i in range(num_sim))
                    for i, (cert, samples) in enumerate(results_drce):
                        drce_cert[i] = cert
                        output_drce[i] = samples
                
                    
                
                    J_DRCE_OS_list= []
                    
                    
                    DRCE_prob = np.empty(num_sim)
                    #drce-----------------------
                    for i in range(num_sim):
                        J_DRCE_list = []
                        for out in output_drce[i]:
                            J_DRCE_list.append(out['cost'][0])
                        
                        # Average cost when using drce[i] controller   
                        J_DRCE_mean= np.mean(J_DRCE_list)
                        J_DRCE_OS_list.append(J_DRCE_mean)
                        DRCE_prob[i] = (J_DRCE_mean <= drce_cert[i])
                       
                    
                    output_J_DRCE_mean.append( np.mean(J_DRCE_OS_list) )
                    output_J_DRCE_std.append( np.std(J_DRCE_OS_list) )
                    output_DRCE_prob.append(np.mean(DRCE_prob))
                    print(" Out-of-Sample cost (DRCE) : ", np.mean(J_DRCE_OS_list))
                    print(" Prob: ",np.mean(DRCE_prob) )
                    
                    #-------------------------------
                    # Data Save
                    theta_v_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
                    theta_w_ = f"_{str(theta_w).replace('.', '_')}" # change 1.0 to 1_0 for file name
                    
                    save_data(path + 'drce_mean_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', output_J_DRCE_mean)
                    save_data(path + 'drce_std_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', output_J_DRCE_std)
                    
                    
                    save_data(path + 'drce_prob_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', output_DRCE_prob)

                    
                    # -- Lambda Save -- 
                    save_data(path + 'drce_lambda_' + theta_w_ +'.pkl', DRCE_lambda)
                    #-------------------------------
                    print("num_noise_sample : ", num_noise, " / finished with dist : ", dist, "/ noise_dist : ", noise_dist, "/ seed : ", seed)
                
    print("Data generation Completed!!")
    print("To get a plot, Use : python plot_osp.py")
    

    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    parser.add_argument('--num_sim', required=False, default=100, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=15, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=15, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=20, type=int) #horizon length
    
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon)
