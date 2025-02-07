#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

class LQG:
    def __init__(self, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat):
        self.dist = dist
        self.noise_dist = noise_dist
        self.T = T
        self.A, self.B, self.C, self.Q, self.Qf, self.R, self.M = system_data
        self.v_mean_hat = v_mean_hat
        self.M_hat = M_hat
        self.nx = self.B.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        self.x0_mean = x0_mean
        self.x0_cov = x0_cov
        self.x0_mean_hat = x0_mean_hat
        self.x0_cov_hat = x0_cov_hat
        self.mu_hat = mu_hat
        self.Sigma_hat = Sigma_hat 
        self.mu_w = mu_w
        self.mu_v = mu_v
        self.Sigma_w = Sigma_w
        
        
        
        if self.dist=="uniform" or self.dist=="quadratic":
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min

        if self.noise_dist =="uniform" or self.noise_dist =="quadratic":
            self.v_max = v_max
            self.v_min = v_min

        #Initial state
            
        self.J = np.zeros(self.T+1)
        self.P = np.zeros((self.T+1, self.nx, self.nx))
        self.S = np.zeros((self.T+1, self.nx, self.nx))
        self.r = np.zeros((self.T+1, self.nx, 1))
        self.z = np.zeros(self.T+1)
        self.K = np.zeros(( self.T, self.nu, self.nx))
        self.L = np.zeros(( self.T, self.nu, 1))


    def uniform(self, a, b, N=1):
        n = a.shape[0]
        x = a + (b-a)*np.random.rand(N,n)
        return x.T

    def normal(self, mu, Sigma, N=1):
        n = mu.shape[0]
        w = np.random.normal(size=(N,n))
        if (Sigma == 0).all():
            x = mu
        else:
            x = mu + np.linalg.cholesky(Sigma) @ w.T
        return x
    
    def quad_inverse(self, x, b, a):
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
    def quadratic(self, wmax, wmin, N=1):
        n = wmin.shape[0]
        x = np.random.rand(N, n)
        x = self.quad_inverse(x, wmax, wmin)
        return x.T
    
    def kalman_filter_cov(self, M_hat, P, P_w=None):
        #Performs state estimation based on the current state estimate, control input and new observation
        if P_w is None:
            #Initial state estimate
            P_ = P
        else:
            #Prediction update
            P_ = self.A @ P @ self.A.T + P_w

        #Measurement update
        temp = np.linalg.solve(self.C @ P_ @ self.C.T + M_hat, self.C @ P_)
        P_new = P_ - P_ @ self.C.T @ temp
        return P_new
    
    def kalman_filter(self, v_mean_hat, M_hat, x, P, y, mu_w=None, u = None):
        #Performs state estimation based on the current state estimate, control input and new observation
        if u is None:
            #Initial state estimate
            x_ = x
        else:
            #Prediction update
            x_ = self.A @ x + self.B @ u + mu_w
            
        P_ = self.C @ P @ self.C.T + M_hat
        #Measurement update
        resid = y - (self.C @ x_ + v_mean_hat)

        temp = np.linalg.solve(M_hat, resid)
        x_new = x_ + P @ self.C.T @ temp
        return x_new

    def riccati(self, Phi, P, S, r, z, Sigma_hat, mu_hat):
        #Riccati equation for standard LQG

        temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
        P_ = self.Q + self.A.T @ temp @ P @ self.A
        S_ = self.Q + self.A.T @ (P + S) @ self.A - P_
        r_ = self.A.T @ temp @ (r + P @ mu_hat)
        z_ = z + np.trace((S + P) @ Sigma_hat) \
            + (2*mu_hat - Phi @ r).T @ temp @ r + mu_hat.T @ temp @ P @ mu_hat
        temp2 = np.linalg.solve(self.R, self.B.T)
        K = - temp2 @ temp @ P @ self.A
        L = - temp2 @ temp @ (r + P @ mu_hat)
        return P_, S_, r_, z_, K, L

    def get_obs(self, x, v):
        #Get new noisy observation
        obs = self.C @ x + v
        return obs

    def backward(self):
        offline_start = time.time()
        #Compute P, S, r, z, K and L backward in time

        self.P[self.T] = self.Qf
        Phi = self.B @ np.linalg.inv(self.R) @ self.B.T
        for t in range(self.T-1, -1, -1):
             self.P[t], self.S[t], self.r[t], self.z[t], self.K[t], self.L[t]  = self.riccati(Phi, self.P[t+1], self.S[t+1], self.r[t+1], self.z[t+1], self.Sigma_hat[t], self.mu_hat[t])
        
        self.x_cov = np.zeros((self.T+1, self.nx, self.nx))
        self.x_cov[0] = self.kalman_filter_cov(self.M_hat[0], self.x0_cov_hat)
        for t in range(self.T):
            self.x_cov[t+1] = self.kalman_filter_cov(self.M_hat[t+1], self.x_cov[t], self.Sigma_hat[t])
        
        offline_end = time.time()
        self.offline_time = offline_end - offline_start # time consumed for offline process
    
    
            
    def forward(self):
        #Apply the controller forward in time.
        start = time.time()
        x = np.zeros((self.T+1, self.nx, 1))
        y = np.zeros((self.T+1, self.ny, 1))
        u = np.zeros((self.T, self.nu, 1))

        x_mean = np.zeros((self.T+1, self.nx, 1))
        J = np.zeros(self.T+1)
        #---system----
        if self.dist=="normal":
            x[0] = self.normal(self.x0_mean, self.x0_cov)
        elif self.dist=="uniform":
            x[0] = self.uniform(self.x0_max, self.x0_min)
        elif self.dist=="quadratic":
            x[0] = self.quadratic(self.x0_max, self.x0_min)
        #---noise----
        if self.noise_dist=="normal":
            true_v = self.normal(self.mu_v, self.M) #observation noise
        elif self.noise_dist=="uniform":
            true_v = self.uniform(self.v_max, self.v_min) #observation noise
        elif self.noise_dist=="quadratic":
            true_v = self.quadratic(self.v_max, self.v_min) #observation noise
            
        y[0] = self.get_obs(x[0], true_v) #initial observation
        x_mean[0] = self.kalman_filter(self.v_mean_hat[0] , self.M_hat[0], self.x0_mean_hat, self.x_cov[0], y[0]) #initial state estimation
        
        for t in range(self.T):
            #disturbance sampling
            if self.dist=="normal":
                true_w = self.normal(self.mu_w, self.Sigma_w)
            elif self.dist=="uniform":
                true_w = self.uniform(self.w_max, self.w_min)
            elif self.dist=="quadratic":
                true_w = self.quadratic(self.w_max, self.w_min)
            #noise sampling
            if self.noise_dist=="normal":
                true_v = self.normal(self.mu_v, self.M) #observation noise
            elif self.noise_dist=="uniform":
                true_v = self.uniform(self.v_max, self.v_min) #observation noise
            elif self.noise_dist=="quadratic":
                true_v = self.quadratic(self.v_max, self.v_min) #observation noise
            #Apply the control input to the system
            u[t] = self.K[t] @ x_mean[t] + self.L[t]
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w
            y[t+1] = self.get_obs(x[t+1], true_v)

            #Update the state estimation (using the nominal mean and covariance)
            x_mean[t+1] = self.kalman_filter(self.v_mean_hat[t+1], self.M_hat[t+1], x_mean[t], self.x_cov[t+1], y[t+1], self.mu_hat[t], u=u[t])
       
       
        self.J_mse = np.zeros(self.T + 1) # State estimation error MSE

        
        #Collect Estimation MSE 
        for t in range(self.T):
            #print("LQG S[",t,"] : ", np.linalg.norm(self.S[t]))
            self.J_mse[t] = (x_mean[t]-x[t]).T@(self.S[t])@(x_mean[t]-x[t])
        
        #Compute the total cost
        J[self.T] = x[self.T].T @ self.Qf @ x[self.T]
        for t in range(self.T-1, -1, -1):
            J[t] = J[t+1] + x[t].T @ self.Q @ x[t] + u[t].T @ self.R @ u[t]
        end = time.time()
        time_ = end-start
        return {'comp_time': time_,
                'state_traj': x,
                'output_traj': y,
                'control_traj': u,
                'cost': J,
                'mse':self.J_mse,
                'offline_time':self.offline_time}
        
    def forward_track(self, desired_trajectory):
        #Apply the controller forward in time.
        start = time.time()
        x = np.zeros((self.T+1, self.nx, 1))
        y = np.zeros((self.T+1, self.ny, 1))
        u = np.zeros((self.T, self.nu, 1))
        error = np.zeros((self.T+1, self.nx, 1)) # Tracking error
        
        x_mean = np.zeros((self.T+1, self.nx, 1))
        J = np.zeros(self.T+1)
        #---system----
        if self.dist=="normal":
            x[0] = self.normal(self.x0_mean, self.x0_cov)
        elif self.dist=="uniform":
            x[0] = self.uniform(self.x0_max, self.x0_min)
        elif self.dist=="quadratic":
            x[0] = self.quadratic(self.x0_max, self.x0_min)
        #---noise----
        if self.noise_dist=="normal":
            true_v = self.normal(self.mu_v, self.M) #observation noise
        elif self.noise_dist=="uniform":
            true_v = self.uniform(self.v_max, self.v_min) #observation noise
        elif self.noise_dist=="quadratic":
            true_v = self.quadratic(self.v_max, self.v_min) #observation noise
            
        y[0] = self.get_obs(x[0], true_v) #initial observation
        x_mean[0] = self.kalman_filter(self.v_mean_hat[0] , self.M_hat[0], self.x0_mean_hat, self.x_cov[0], y[0]) #initial state estimation
        
        for t in range(self.T):
            #disturbance sampling
            if self.dist=="normal":
                true_w = self.normal(self.mu_w, self.Sigma_w)
            elif self.dist=="uniform":
                true_w = self.uniform(self.w_max, self.w_min)
            elif self.dist=="quadratic":
                true_w = self.quadratic(self.w_max, self.w_min)
            #noise sampling
            if self.noise_dist=="normal":
                true_v = self.normal(self.mu_v, self.M) #observation noise
            elif self.noise_dist=="uniform":
                true_v = self.uniform(self.v_max, self.v_min) #observation noise
            elif self.noise_dist=="quadratic":
                true_v = self.quadratic(self.v_max, self.v_min) #observation noise
                
            # Get desired trajectory at current time step (desired position and velocity)
            traj = desired_trajectory[:,t].reshape(-1, 1)
            #print("traj[t] : ", traj)
            #print("current[t] : ",x_mean[t])
            error[t] = x_mean[t] - traj  # Error as a 4D vector
            
            
            #Apply the control input to the system
            #u[t] = self.K[t] @ x_mean[t] + self.L[t]
            u[t] = self.K[t] @ error[t] + self.L[t]
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w
            y[t+1] = self.get_obs(x[t+1], true_v)

            #Update the state estimation (using the nominal mean and covariance)
            x_mean[t+1] = self.kalman_filter(self.v_mean_hat[t+1], self.M_hat[t+1], x_mean[t], self.x_cov[t+1], y[t+1], self.mu_hat[t], u=u[t])
       
       
        self.J_mse = np.zeros(self.T + 1) # State estimation error MSE

        
        #Collect Estimation MSE 
        for t in range(self.T):
            #print("LQG S[",t,"] : ", np.linalg.norm(self.S[t]))
            self.J_mse[t] = (x_mean[t]-x[t]).T@(self.S[t])@(x_mean[t]-x[t])
        
        #Compute the total cost
        J[self.T] = (x[self.T-1]-desired_trajectory[:,self.T-1].reshape(-1, 1)).T @ self.Qf @ (x[self.T-1]-desired_trajectory[:,self.T-1].reshape(-1, 1))
        for t in range(self.T-2, -1, -1):
            J[t] = J[t+1] + (x[t]-desired_trajectory[:,t].reshape(-1, 1)).T @ self.Q @ (x[t]-desired_trajectory[:,t].reshape(-1, 1)) + u[t].T @ self.R @ u[t]
            
        end = time.time()
        time_ = end-start
        return {'comp_time': time_,
                'state_traj': x,
                'output_traj': y,
                'control_traj': u,
                'cost': J,
                'mse':self.J_mse,
                'offline_time':self.offline_time,
                'desired_traj': desired_trajectory}