#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
import cvxpy as cp
import scipy
from joblib import Parallel, delayed

# Distributionally Robust regret-optimal with measurement feedback (DR-RO-MF)
class DR_RO_MF:
    def __init__(self, theta, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat):
        self.dist = dist
        self.noise_dist = noise_dist
        self.T = T
        self.A, self.B, self.C, self.Q, self.Qf, self.R, self.M = system_data
        self.v_mean_hat = v_mean_hat 
        self.M_hat = M_hat 
        self.nx = self.B.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        self.x0_mean = x0_mean # true
        self.x0_cov = x0_cov # true
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
        
        #---system----
        if self.dist=="normal":
            self.x0_init = self.normal(self.x0_mean, self.x0_cov)
        elif self.dist=="uniform":
            self.x0_init = self.uniform(self.x0_max, self.x0_min)
        elif self.dist=="quadratic":
            self.x0_init = self.quadratic(self.x0_max, self.x0_min)
        #---noise----
        if self.noise_dist=="normal":
            self.true_v_init = self.normal(self.mu_v, self.M) #observation noise
        elif self.noise_dist=="uniform":
            self.true_v_init = self.uniform(self.v_max, self.v_min) #observation noise
        elif self.noise_dist=="quadratic":
            self.true_v_init = self.quadratic(self.v_max, self.v_min) #observation noise
        
        self.theta_w = theta # size of the ambiguity set
        
        self.DR_sdp = self.create_DR_sdp()
        
        print("DR_RO_MF")
            

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
    
    
    def create_DR_sdp(self):
        
        
        Y = cp.Variable((self.T * self.nu, self.T * self.ny), name='Y')
        X = cp.Variable((self.T * (self.nx + self.ny), self.T * (self.nx + self.ny)), symmetric = True, name='X')
        X11 = cp.Variable((self.T * self.nx, self.T * self.nx))
        X12 = cp.Variable((self.T * self.nx, self.T * self.ny))
        X22 = cp.Variable((self.T * self.ny, self.T * self.ny))
        
        gamma = cp.Parameter(1)
        radius = cp.Parameter(1)
        
        I_nx = cp.Constant(np.eye(self.T * self.nx))
        I_nu = cp.Constant(np.eye(self.T * self.nu))
        I_ny = cp.Constant(np.eye(self.T * self.ny))
        
        Zero_xy = cp.Constant(np.zeros((self.T * self.nx, self.T * self.ny)))
        Zero_xu = cp.Constant(np.zeros((self.T * self.nx, self.T * self.nu)))
        Zero_yu = cp.Constant(np.zeros((self.T * self.ny, self.T * self.nu)))
        
        obj = gamma * (radius**2 - self.T * (self.nx + self.ny)) + cp.trace(X)
        
        constraints=[
            Y == cp.tril(Y), # Y is lower triangular 
            X >> 0,
            X == cp.bmat([[X11, X12],
                          [X12.T, X22]]),
            cp.bmat([[X11, X12, gamma * I_nx, Zero_xy, Zero_xu],
                    [X12.T, X22, Zero_xy.T, gamma * I_ny, Zero_yu],
                    [gamma * I_nx, Zero_xy, gamma * I_nx, ]
                    ]) >> 0
        ]
        # V = cp.Variable((self.nx, self.nx), symmetric=True, name='V')
        # Sigma_wc = cp.Variable((self.nx, self.nx), symmetric=True, name='Sigma_wc')
        # Y = cp.Variable((self.nx,self.nx), name='Y')
        # X_pred = cp.Variable((self.nx,self.nx), symmetric=True, name='X_pred')
        # M_test = cp.Variable((self.ny, self.ny), symmetric=True, name='M_test')
        # Z = cp.Variable((self.ny, self.ny), name='Z')
        
        # #Parameters
        # S_var = cp.Parameter((self.nx,self.nx), name='S_var')
        # P_var = cp.Parameter((self.nx,self.nx), name='P_var')
        # Lambda_ = cp.Parameter(1, name='Lambda_')
        # Sigma_w = cp.Parameter((self.nx, self.nx), name='Sigma_w') # nominal Sigma_w
        # x_cov = cp.Parameter((self.nx, self.nx), name='x_cov') # x_cov from before time step
        # M_hat = cp.Parameter((self.ny, self.ny), name='M_hat')
        # radi = cp.Parameter(nonneg=True, name='radi')
               
        # #use Schur Complements
        # #obj function
        # obj = cp.Maximize(cp.trace(S_var @ V) + cp.trace((P_var - Lambda_ * np.eye(self.nx)) @ Sigma_wc) + 2*Lambda_*cp.trace(Y)) 
        
        # #constraints
        # constraints = [
        #         cp.bmat([[Sigma_w, Y],
        #                  [Y.T, Sigma_wc]
        #                  ]) >> 0,
        #         X_pred == self.A @ x_cov @ self.A.T + Sigma_wc,
        #         cp.bmat([[X_pred-V, X_pred @ self.C.T],
        #                 [self.C @ X_pred, self.C @ X_pred @ self.C.T + M_test]
        #                 ]) >> 0 ,
        #         self.C @ X_pred @ self.C.T + M_test >>0,
        #         cp.trace(M_hat + M_test - 2*Z ) <= radi**2,
        #         cp.bmat([[M_hat, Z],
        #                  [Z.T, M_test]
        #                  ]) >> 0,
        #         V>>0,
        #         X_pred >>0,
        #         M_test >>0,
        #         Sigma_wc >>0
        #         ]
        
        # prob = cp.Problem(obj, constraints)
        return prob
        
    def solve_DR_sdp(self, prob, P_t1, S_t1, M, X_cov, Sigma_hat, theta, Lambda_):
        #construct problem
        params = prob.parameters()
        # for i, param in enumerate(params):
        #     print(f"params[{i}]: {param.name()}")
        params[0].value = S_t1 # S[t+1]
        params[1].value = P_t1 # P[t+1]
        params[2].value = Lambda_
        params[3].value = Sigma_hat
        params[4].value = X_cov
        params[5].value = M # Noise covariance
        params[6].value = theta
        
        
        prob.solve(solver=cp.MOSEK)
        # if prob.status in ["infeasible", "unbounded"]:
        #     print("theta_w : ", self.theta_w, " theta_v : ", self.theta_v)
        #     print("Lambda_:" , Lambda_)
        #     print(prob.status, 'False in DRCE SDP !!!!!!!!')
        
        sol = prob.variables()
        # for i, var in enumerate(sol):
        #     print(f"var[{i}]: {var.name()}")
        #['V', 'Sigma_wc', 'Y', 'X_pred', 'M_test', 'Z']
        
        S_xx_opt = sol[3].value
        #S_xy_opt = S_xx_opt @ self.C.T
        #S_yy_opt = self.C @ S_xx_opt @ self.C.T + sol[4].value
        
        M_wc_opt = sol[4].value
        Sigma_wc_opt = sol[1].value
        cost = prob.value
        return  S_xx_opt,  Sigma_wc_opt, M_wc_opt, cost, prob.status
    
    

    def get_obs(self, x, v):
        #Get new noisy observation
        obs = self.C @ x + v
        return obs

    def backward(self):
        offline_start = time.time()
        self.P[self.T] = self.Qf
        if self.lambda_ <= np.max(np.linalg.eigvals(self.P[self.T])) or self.lambda_<= np.max(np.linalg.eigvals(self.P[self.T] + self.S[self.T])):
            print("t={}: False!".format(self.T))

        Phi = self.B @ np.linalg.inv(self.R) @ self.B.T + 1/self.lambda_ * np.eye(self.nx)
        for t in range(self.T-1, -1, -1):
            self.P[t], self.S[t], self.r[t], self.z[t], self.K[t], self.L[t], self.H[t], self.h[t], self.g[t] = self.riccati(Phi, self.P[t+1], self.S[t+1], self.r[t+1], self.z[t+1], self.Sigma_hat[t], self.mu_hat[t], self.lambda_, t)

        self.x_cov = np.zeros((self.T+1, self.nx, self.nx))
        self.S_opt = np.zeros((self.T+1, self.nx + self.ny, self.nx + self.ny))
        self.S_xx = np.zeros((self.T+1, self.nx, self.nx))
        self.S_xy = np.zeros((self.T+1, self.nx, self.ny))
        self.S_yy = np.zeros((self.T+1, self.ny, self.ny))
        self.sigma_wc = np.zeros((self.T, self.nx, self.nx))
        self.x_cov[0], self.S_xx[0], self.S_xy[0], self.S_yy[0], _, status= self.DR_kalman_filter_cov_initial(self.M_hat[0], self.x0_cov_hat, self.S[0])
        
        for t in range(self.T):
            print("DRCE Offline step : ",t,"/",self.T)
            self.x_cov[t+1], self.S_xx[t+1], self.S_xy[t+1], self.S_yy[t+1], self.sigma_wc[t], _, status= self.DR_kalman_filter_cov(self.P[t+1], self.S[t+1], self.M_hat[t+1], self.x_cov[t], self.Sigma_hat[t], self.lambda_)
        
        offline_end = time.time()
        self.offline_time = offline_end - offline_start

    def forward(self):
        #Apply the controller forward in time.
        start = time.time()
        x = np.zeros((self.T+1, self.nx, 1))
        y = np.zeros((self.T+1, self.ny, 1))
        u = np.zeros((self.T, self.nu, 1))

        x_mean = np.zeros((self.T+1, self.nx, 1))
        J = np.zeros(self.T+1)
        mu_wc = np.zeros((self.T, self.nx, 1))

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
        
        x_mean[0] = self.DR_kalman_filter(self.v_mean_hat[0], self.M_hat[0], self.x0_mean_hat, y[0], self.S_xx[0], self.S_xy[0], self.S_yy[0]) #initial state estimation

        for t in range(self.T):
            mu_wc[t] = self.H[t] @ x_mean[t] + self.h[t] #worst-case mean
            
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

            #Update the state estimation (using the worst-case mean and covariance)
            x_mean[t+1] = self.DR_kalman_filter(self.v_mean_hat[t+1], self.M_hat[t+1], x_mean[t], y[t+1], self.S_xx[t+1], self.S_xy[t+1], self.S_yy[t+1], mu_wc[t], u=u[t])

        #State estimation error MSE
        self.J_mse = np.zeros(self.T + 1) 
        #Collect Estimation MSE 
        for t in range(self.T):
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
        mu_wc = np.zeros((self.T, self.nx, 1))

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
        
        x_mean[0] = self.DR_kalman_filter(self.v_mean_hat[0], self.M_hat[0], self.x0_mean_hat, y[0], self.S_xx[0], self.S_xy[0], self.S_yy[0]) #initial state estimation

        for t in range(self.T):
            mu_wc[t] = self.H[t] @ x_mean[t] + self.h[t] #worst-case mean
            
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
            error[t] = x_mean[t] - traj  # Error as a 4D vector
            
            #Apply the control input to the system
            u[t] = self.K[t] @ error[t] + self.L[t]
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w 
            y[t+1] = self.get_obs(x[t+1], true_v)

            #Update the state estimation (using the worst-case mean and covariance)
            x_mean[t+1] = self.DR_kalman_filter(self.v_mean_hat[t+1], self.M_hat[t+1], x_mean[t], y[t+1], self.S_xx[t+1], self.S_xy[t+1], self.S_yy[t+1], mu_wc[t], u=u[t])

        #State estimation error MSE
        self.J_mse = np.zeros(self.T + 1) 
        #Collect Estimation MSE 
        for t in range(self.T):
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

