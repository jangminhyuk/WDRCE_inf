#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.DRCE import DRCE
from controllers.DRLQC import DRLQC
from joblib import Parallel, delayed
from scipy.linalg import expm
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


def save_pickle_data(path, data):
    """Save data using pickle to the specified path."""
    with open(path, 'wb') as output:
        pickle.dump(data, output)

def load_pickle_data(path):
    """Load data using pickle from the specified path."""
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)
    
def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()
def create_matrices(nx, ny, nu):
    A = np.load("./inputs/power/A.npy") # (n x n) matrix
    B = np.load("./inputs/power/B.npy")
    C = np.hstack([np.eye(ny, int(ny/2)), np.zeros((ny, int((nx-ny)/2))), np.eye(ny, int(ny/2), k=-int(ny/2)), np.zeros((ny, int((nx-ny)/2)))])
    return A, B ,C
def main(dist, noise_dist, num_sim, num_samples, num_noise_samples, T_total, trajectory):
    
    lambda_ = 10
    seed = 2024 # Random seed
    np.random.seed(seed)
    # --- ----- --------#
    num_noise_list = [num_noise_samples]
    
    if dist=='normal' and trajectory=='curvy':
        theta_v_list = [1.0, 2.0, 3.0, 4.0, 5.0] # radius of noise ambiguity set
        theta_w_list = [0.1]
        lambda_list = [10000, 15000, 20000, 25000, 30000, 35000, 40000]
    elif dist=='normal' and trajectory=='circular':
        theta_v_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        theta_w_list = [0.1]
        lambda_list = [10000, 15000, 20000, 25000, 30000, 35000, 40000]
        
    num_x0_samples = 15 #  N_x0 
    theta_x0 = 0.5 # radius of initial state ambiguity set
    
    use_lambda = True # If use_lambda=True, we will use lambda_list. If use_lambda=False, we will use theta_w_list.
    use_optimal_lambda = False
    if use_lambda:
        dist_parameter_list = lambda_list
    else:
        dist_parameter_list = theta_w_list
    
    output_J_LQG_mean, output_J_WDRC_mean, output_J_DRCE_mean, output_J_DRLQC_mean =[], [], [], []
    output_J_LQG_std, output_J_WDRC_std, output_J_DRCE_std, output_J_DRLQC_std=[], [], [], []
    #-------Initialization-------
    # Constants
    Amp = 5.0  # Amplitude of the sine wave (for curvy trajectory)
    slope = 1.0  # Slope of the linear y trajectory (for curvy trajectory)
    radius = 5.0  # Radius of the circle (for circular trajectory)
    omega = 0.5  # Angular frequency
    
    nx = 4
    nu = 2
    ny = 2
    # Time for simulation
    dt = 0.1  # 
    time_steps = int(T_total / dt) + 1  # Number of simulation steps
    time = np.linspace(0, T_total, time_steps)  # Time vector

    # Discrete-time system dynamics (for the 4D double integrator: x, y positions and velocities)
    A = np.array([[1, dt, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, dt],
                    [0, 0, 0, 1]])

    B = np.array([[0.5 * dt**2, 0],
                    [dt, 0],
                    [0, 0.5 * dt**2],
                    [0, dt]])
    C = np.array([[0, 1, 0, 0],
              [0, 0, 0, 1]])
    Q = np.eye(4) * np.array([20, 1, 20, 1])  # Heavily penalize position errors, less for velocity errors
    Qf = np.eye(4) * np.array([50, 1, 50, 1])
    R = 0.01 * np.eye(2)  # Control cost for both x and y accelerations
    
    # Initialize arrays to store desired trajectories
    x_d = np.zeros(time_steps)
    vx_d = np.zeros(time_steps)
    y_d = np.zeros(time_steps)
    vy_d = np.zeros(time_steps)
    # Desired trajectory function based on the command-line argument
    if trajectory == 'curvy':
        # Curvy trajectory: sine wave
        x_d = Amp * np.sin(omega * time) 
        vx_d = Amp * omega * np.cos(omega * time) 
        y_d = slope * time 
        vy_d = slope * np.ones(time_steps)
    elif trajectory == 'circular':
        # Circular trajectory: circular motion for both x and y
        x_d = radius * np.cos(omega * time)
        vx_d = -radius * omega * np.sin(omega * time)
        y_d = radius * np.sin(omega * time)
        vy_d = radius * omega * np.cos(omega * time)
        
    desired_traj = np.vstack((x_d, vx_d, y_d, vy_d))
    # Set the initial state to match the first point of the desired trajectory
    initial_trajectory = desired_traj[:,0].reshape(-1,1) # Get the initial desired state at t=0
    
    #-------Disturbance Distribution-------
    if dist == "normal" and trajectory =='curvy':
        #disturbance distribution parameters
        w_max = None
        w_min = None
        mu_w = 0.02*np.ones((nx, 1))
        Sigma_w= 0.02*np.eye(nx)
        #initial state distribution parameters
        x0_max = None
        x0_min = None
        x0_mean = initial_trajectory
        x0_cov = 0.01*np.eye(nx)
    elif dist == "normal" and trajectory =='circular':
        #disturbance distribution parameters
        w_max = None
        w_min = None
        mu_w = 0.01*np.ones((nx, 1))
        Sigma_w= 0.01*np.eye(nx)
        #initial state distribution parameters
        x0_max = None
        x0_min = None
        x0_mean = initial_trajectory
        x0_cov = 0.01*np.eye(nx)
        
    #-------Noise distribution ---------#
    if noise_dist =="normal":
        v_max = None
        v_min = None
        M = 0.1*np.eye(ny) #observation noise covariance
        mu_v = 0.1*np.ones((ny, 1))

    #-------Estimate the nominal distribution-------
    # Nominal initial state distribution
    x0_mean_hat, x0_cov_hat = gen_sample_dist(dist, 1, num_x0_samples, mu_w=x0_mean, Sigma_w=x0_cov, w_max=x0_max, w_min=x0_min)
    # Nominal Disturbance distribution
    mu_hat, Sigma_hat = gen_sample_dist(dist,time_steps+1, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
    # Nominal Noise distribution
    v_mean_hat, M_hat = gen_sample_dist(noise_dist, time_steps+1, num_noise_samples, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min)

    M_hat = M_hat + 1e-4*np.eye(ny) # to prevent numerical error from inverse in standard KF at small sample size
    Sigma_hat = Sigma_hat + 1e-4*np.eye(nx)
    x0_cov_hat = x0_cov_hat + 1e-4*np.eye(nx) 
    
    # Create paths for saving individual results
    temp_results_path = "./temp_results/"
    if not os.path.exists(temp_results_path):
        os.makedirs(temp_results_path)
    def perform_simulation(lambda_, noise_dist, dist_parameter, theta, idx_w, idx_v):
        for num_noise in num_noise_list:
            np.random.seed(seed) # fix Random seed!
            theta_w = 1.0 # Will be used only when if use_lambda = True, this value will be in DRLQC method. (Since WDRC and DRCE will use lambdas)
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
                path = "./results/{}_{}/finite/multiple/params_lambda/2d_{}/".format(dist, noise_dist,trajectory)
            else:
                path = "./results/{}_{}/finite/multiple/params_thetas/2d_{}/".format(dist, noise_dist,trajectory)
                
            if not os.path.exists(path):
                os.makedirs(path)
        
            
            #-------Create a random system-------
            system_data = (A, B, C, Q, Qf, R, M)
            
            #-------Perform n independent simulations and summarize the results-------
            output_lqg_list = []
            output_wdrc_list = []
            output_drce_list = []
            output_drlqc_list = []
            
            #-----Initialize controllers-----
            wdrc = WDRC(lambda_, theta_w, time_steps, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda, use_optimal_lambda)
            drce = DRCE(lambda_, theta_w, theta, theta_x0, time_steps, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat,  M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda, use_optimal_lambda)
            lqg = LQG(time_steps, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat , x0_mean_hat[0], x0_cov_hat[0])
        
            wdrc.backward()
            drce.backward()
            lqg.backward()
            
            print('---------------------')
            np.random.seed(seed) # fix Random seed!
            #----------------------------
            print("Running DRCE Forward step ...")
            for i in range(num_sim):
                
                #Perform state estimation and apply the controller
                output_drce = drce.forward_track(desired_traj)
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
            np.random.seed(seed) # fix Random seed!
            
            #----------------------------             
            np.random.seed(seed) # fix Random seed!
            print("Running WDRC Forward step ...")  
            for i in range(num_sim):
        
                #Perform state estimation and apply the controller
                output_wdrc = wdrc.forward_track(desired_traj)
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
            print("Running LQG Forward step ...")
            for i in range(num_sim):
                output_lqg = lqg.forward_track(desired_traj)
                output_lqg_list.append(output_lqg)
        
                if i%50==0:
                    print("Simulation #",i, ' | cost (LQG):', output_lqg['cost'][0], 'time (LQG):', output_lqg['comp_time'])
                
            J_LQG_list = []
            for out in output_lqg_list:
                J_LQG_list.append(out['cost'])
            J_LQG_mean= np.mean(J_LQG_list, axis=0)
            J_LQG_std = np.std(J_LQG_list, axis=0)
            output_J_LQG_mean.append(J_LQG_mean[0])
            output_J_LQG_std.append(J_LQG_std[0])
            print(" Average cost (LQG) : ", J_LQG_mean[0])
            print(" std (LQG) : ", J_LQG_std[0])
            
            #-----------------------------------------
            # Save data #
            theta_v_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
            theta_w_ = f"_{str(theta_w).replace('.', '_')}" # change 1.0 to 1_0 for file name
            #save_data(path + 'drlqc' + theta_w_ + 'and' + theta_v_+ '.pkl', J_DRLQC_mean)
            if use_lambda:
                save_data(path + 'drce_' +  str(lambda_) + 'and' + theta_v_+ '.pkl', J_DRCE_mean)
                save_data(path + 'wdrc_' + str(lambda_) + '.pkl', J_WDRC_mean)
            else:
                save_data(path + 'drce' + theta_w_ + 'and' + theta_v_+ '.pkl', J_DRCE_mean)
                save_data(path + 'wdrc' + theta_w_ + '.pkl', J_WDRC_mean)
                
            save_data(path + 'lqg.pkl', J_LQG_mean)
            #Summarize and plot the results
            #Save all raw data
            if use_lambda:
                rawpath = "./results/{}_{}/finite/multiple/params_lambda/2d_{}/raw/".format(dist, noise_dist, trajectory)
            else:
                rawpath = "./results/{}_{}/finite/multiple/params_thetas/2d_{}/raw/".format(dist, noise_dist, trajectory)
                
            if not os.path.exists(rawpath):
                os.makedirs(rawpath)
                
            if use_lambda:
                save_data(rawpath + 'drce_' + str(lambda_) + 'and' + theta_v_+ '.pkl', output_drce_list)
                save_data(rawpath + 'wdrc_' + str(lambda_) + '.pkl', output_wdrc_list)
            else:
                save_data(rawpath + 'drce_' + theta_w_ + 'and' + theta_v_+ '.pkl', output_drlqc_list)
                save_data(rawpath + 'wdrc' + theta_w_ + '.pkl', output_wdrc_list)
                
            save_data(rawpath + 'lqg.pkl', output_lqg_list)
            print('\n-------Summary-------')
            print("dist : ", dist,"/ noise dist : ", noise_dist, "/ num_samples : ", num_samples, "/ num_noise_samples : ", num_noise, "/seed : ", seed)
    
    combinations = [(dist_parameter, theta, idx_w, idx_v) for idx_w, dist_parameter in enumerate(dist_parameter_list) for idx_v, theta in enumerate(theta_v_list)]
    
    results = Parallel(n_jobs=-1)(
                delayed(perform_simulation)(lambda_, noise_dist, dist_parameter, theta, idx_w, idx_v)
                for dist_parameter, theta, idx_w, idx_v in combinations
            )     
            
    print("Params data generation Completed !")
    print("Please make sure your lambda_list(or theta_w_list) and theta_v_list in plot file is as desired")
    if use_lambda:
        print("Now use : python plot4_2d.py --use_lambda --dist "+ dist + " --noise_dist " + noise_dist+ " --trajectory "+trajectory)
        print("Now use : python plot_params_2D.py --use_lambda --dist "+ dist + " --noise_dist " + noise_dist+ " --trajectory "+trajectory)
    else:
        print("Now use : python plot4_2d.py --dist "+ dist + " --noise_dist " + noise_dist+ " --trajectory "+trajectory)
    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or quadratic)
    parser.add_argument('--num_sim', required=False, default=500, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=15, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=15, type=int) #number of noise samples
    parser.add_argument('--time', required=False, default=20, type=int) #simlation time
    parser.add_argument('--trajectory', required=False, default="curvy", type=str) #desired trajectory
    
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.time, args.trajectory)
