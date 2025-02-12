#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
import cvxpy as cp
import scipy
from joblib import Parallel, delayed

# WDRC algorithm : Downloaded from the authors github
class WDRC:
    def __init__(self, lambda_, theta_w, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat, use_lambda, use_optimal_lambda):
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
        self.theta_w = theta_w
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
            
        print("WDRC ", self.dist, " / ", self.noise_dist)    
        self.sdp_prob = self.gen_sdp()

        if use_lambda==True or use_optimal_lambda==True: # Use pre-computed lambda
            self.lambda_ = np.array([lambda_])
        else:
            self.lambda_ = self.optimize_penalty() #optimize penalty parameter for theta
            
        
        
        self.P = np.zeros((self.T+1, self.nx, self.nx))
        self.S = np.zeros((self.T+1, self.nx, self.nx))
        self.r = np.zeros((self.T+1, self.nx, 1))
        self.z = np.zeros(self.T+1)
        self.K = np.zeros(( self.T, self.nu, self.nx))
        self.L = np.zeros(( self.T, self.nu, 1))
        self.H = np.zeros(( self.T, self.nx, self.nx))
        self.h = np.zeros(( self.T, self.nx, 1))
        self.g = np.zeros(( self.T, self.nx, self.nx))
        
    
    def optimize_penalty(self):
        # Find inf_penalty (infimum value of penalty coefficient satisfying Assumption 1)
        self.infimum_penalty = self.binarysearch_infimum_penalty_finite()
        print("Infimum penalty:", self.infimum_penalty)
        
        
        #Optimize penalty
        print("Optimizing lambda . . . Please wait")
        output = minimize(self.objective, x0=np.array([4*self.infimum_penalty]), method='L-BFGS-B', options={'disp': False, 'maxiter': 1000,'ftol': 1e-6,'gtol': 1e-6, 'maxfun':1000})
        optimal_penalty = output.x
        print("WDRC Optimal penalty (lambda_star) :", optimal_penalty[0], " when theta_w : ", self.theta_w, "\n\n")
        return optimal_penalty


    def objective(self, penalty):
        #Compute the upper bound in Proposition 1
        P = np.zeros((self.T+1, self.nx,self.nx))        
        S = np.zeros((self.T+1, self.nx,self.nx))
        r = np.zeros((self.T+1, self.nx,1))
        z = np.zeros((self.T+1, 1))
        z_tilde = np.zeros((self.T+1, 1))

        if np.max(np.linalg.eigvals(P)) > penalty:
                return np.inf
        if penalty < 0:
            return np.inf
        P[self.T] = self.Qf
        if np.max(np.linalg.eigvals(P[self.T])) > penalty:
                return np.inf
        for t in range(self.T-1, -1, -1):
            Phi = self.B @ np.linalg.inv(self.R) @ self.B.T + 1/penalty * np.eye(self.nx)
            P[t], S[t], r[t], z[t], K, L, H, h, g = self.riccati(Phi, P[t+1], S[t+1], r[t+1], z[t+1], self.Sigma_hat[t], self.mu_hat[t], penalty, t)
            if np.max(np.linalg.eigvals(P[t])) > penalty:
                return np.inf
        
        x_cov = np.zeros((self.T, self.nx, self.nx))
        sigma_wc = np.zeros((self.T, self.nx, self.nx))
        x_cov[0] = self.kalman_filter_cov(self.M_hat[0], self.x0_cov_hat)
        for t in range(0, self.T-1):
            x_cov[t+1] = self.kalman_filter_cov(self.M_hat[t], x_cov[t], sigma_wc[t])
            sigma_wc[t], z_tilde[t], status = self.solve_sdp(self.sdp_prob, penalty, self.M_hat[t], x_cov[t], P[t+1], S[t+1], self.Sigma_hat[t])
            if status in ["infeasible", "unbounded"]:
                print(status)
                print("penalty : ", penalty)
                return np.inf
        obj_val = penalty*self.T*self.theta_w**2 + (self.x0_mean_hat.T @ P[0] @ self.x0_mean_hat)[0][0] + 2*(r[0].T @ self.x0_mean_hat)[0][0] + z[0][0] + np.trace(P[0] @ self.x0_cov_hat)  + np.trace(S[0] @ x_cov[0]) + z_tilde.sum()
        return obj_val/self.T
    
    def binarysearch_infimum_penalty_finite(self):
        left = 0
        right = 100000
        while right - left > 1e-6:
            mid = (left + right) / 2.0
            if self.check_assumption(mid):
                right = mid
            else:
                left = mid
        lam_hat = right
        return lam_hat

    def check_assumption(self, penalty):
        #Check Assumption 1
        P = self.Qf
        S = np.zeros((self.nx,self.nx))
        r = np.zeros((self.nx,1))
        z = np.zeros((1,1))
        if penalty < 0:
            return False
        if np.max(np.linalg.eigvals(P)) >= penalty:
        #or np.max(np.linalg.eigvals(P + S)) >= penalty:
                return False
        for t in range(self.T-1, -1, -1):
            Phi = self.B @ np.linalg.inv(self.R) @ self.B.T + 1/penalty * np.eye(self.nx)
            P, S, r, z, K, L, H, h, g = self.riccati(Phi, P, S, r, z, self.Sigma_hat[t], self.mu_hat[t], penalty, t)
            if np.max(np.linalg.eigvals(P)) >= penalty:
                return False
        return True
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
    
    def gen_sdp(self):
        Sigma = cp.Variable((self.nx,self.nx), symmetric=True)
        Y = cp.Variable((self.nx,self.nx),symmetric = True)
        X = cp.Variable((self.nx,self.nx), symmetric=True)
        X_pred = cp.Variable((self.nx,self.nx), symmetric=True)
    
        P_var = cp.Parameter((self.nx,self.nx))
        lambda_ = cp.Parameter(1)
        S_var = cp.Parameter((self.nx,self.nx))
        #Sigma_hat_12_var = cp.Parameter((self.nx,self.nx))
        Sigma_hat = cp.Parameter((self.nx,self.nx))
        M_hat = cp.Parameter((self.ny,self.ny))
        X_bar = cp.Parameter((self.nx,self.nx))
        
        obj = cp.Maximize(cp.trace((P_var - lambda_*np.eye(self.nx)) @ Sigma) + 2*lambda_*cp.trace(Y) + cp.trace(S_var @ X))
        
        constraints = [
                cp.bmat([[Sigma_hat, Y],
                        [Y.T, Sigma]
                        ]) >> 0,
                # cp.bmat([[Sigma_hat_12_var @ Sigma @ Sigma_hat_12_var, Y],
                #          [Y, np.eye(self.nx)]
                #          ]) >> 0,
                Sigma >> 0,
                X_pred >> 0,
                cp.bmat([[X_pred - X, X_pred @ self.C.T],
                            [self.C @ X_pred, self.C @ X_pred @ self.C.T + M_hat]
                        ]) >> 0,        
                X_pred == self.A @ X_bar @ self.A.T + Sigma,
                self.C @ X_pred @ self.C.T + M_hat >> 0,
                #Y >> 0,
                X >> 0
                ]
        prob = cp.Problem(obj, constraints)
        return prob
        
        
    def solve_sdp(self, sdp_prob, lambda_, M_hat, x_cov, P, S, Sigma_hat):
        params = sdp_prob.parameters()
        params[0].value = P
        params[1].value = lambda_
        params[2].value = S
        params[3].value = Sigma_hat
        params[4].value = M_hat
        params[5].value = x_cov
        
        sdp_prob.solve(solver=cp.MOSEK)
        Sigma = sdp_prob.variables()[0].value
        cost = sdp_prob.value
        status = sdp_prob.status
        return Sigma, cost, status

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

        #Measurement update
        resid = y - (self.C @ x_ + v_mean_hat) 
        temp = np.linalg.solve(M_hat, resid)
        x_new = x_ + P @ self.C.T @ temp
        return x_new

    def riccati(self, Phi, P, S, r, z, Sigma_hat, mu_hat, lambda_, t):
        #Riccati equation corresponding to the Theorem 1

        #temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
        temp = np.linalg.solve(np.eye(self.nx) + P @ Phi, np.eye(self.nx))
        P_ = self.Q + self.A.T @ temp @ P @ self.A
        S_ = self.Q + self.A.T @ P @ self.A - P_

        # Double check Assumption 1
        if lambda_ <= np.max(np.linalg.eigvals(P)):
        #or lambda_ <= np.max(np.linalg.eigvals(P+S)):
            print("t={}: False!!!!!!!!!".format(t))
            print("lambda_: {}".format(lambda_))
            return None
        r_ = self.A.T @ temp @ (r + P @ mu_hat)
        z_ = z - lambda_* np.trace(Sigma_hat) \
                + (2*mu_hat - Phi @ r).T @ temp @ r + mu_hat.T @ temp @ P @ mu_hat
        temp2 = np.linalg.solve(self.R, self.B.T)
        K = - temp2 @ temp @ P @ self.A
        L = - temp2 @ temp @ (r + P @ mu_hat)
        h = np.linalg.inv(lambda_ * np.eye(self.nx) - P) @ (r + P @ self.B @ L + lambda_*mu_hat)
        H = np.linalg.inv(lambda_* np.eye(self.nx)  - P) @ P @ (self.A + self.B @ K)
        g = lambda_**2 * np.linalg.inv(lambda_*np.eye(self.nx) - P) @ Sigma_hat @ np.linalg.inv(lambda_*np.eye(self.nx) - P)
        return P_, S_, r_, z_, K, L, H, h, g

    def get_obs(self, x, v):
        #Get new noisy observation
        obs = self.C @ x + v
        return obs

    def backward(self):
        offline_start = time.time()
        self.P[self.T] = self.Qf
        print("lambda: ",self.lambda_)
        if self.lambda_ <= np.max(np.linalg.eigvals(self.P[self.T])) or self.lambda_<= np.max(np.linalg.eigvals(self.P[self.T] + self.S[self.T])):
            print("t={}: False!".format(self.T))

        Phi = self.B @ np.linalg.inv(self.R) @ self.B.T + 1/self.lambda_ * np.eye(self.nx)
        for t in range(self.T-1, -1, -1):
            self.P[t], self.S[t], self.r[t], self.z[t], self.K[t], self.L[t], self.H[t], self.h[t], self.g[t] = self.riccati(Phi, self.P[t+1], self.S[t+1], self.r[t+1], self.z[t+1], self.Sigma_hat[t], self.mu_hat[t], self.lambda_, t)

        self.x_cov = np.zeros((self.T+1, self.nx, self.nx))
        sigma_wc = np.zeros((self.T, self.nx, self.nx))
        #print(self.v_mean_hat[0])
        self.x_cov[0] = self.kalman_filter_cov(self.M_hat[0], self.x0_cov_hat)
        for t in range(self.T):
            print("WDRC Offline step : ",t,"/",self.T)
            
            sigma_wc[t], _, status = self.solve_sdp(self.sdp_prob, self.lambda_, self.M_hat[t], self.x_cov[t], self.P[t+1], self.S[t+1], self.Sigma_hat[t])
            if status in ["infeasible", "unbounded"]:
                print(status, 'False!!!!!!!!!!!!!', " lambda: ", self.lambda_)
            self.x_cov[t+1] = self.kalman_filter_cov(self.M_hat[t+1], self.x_cov[t], sigma_wc[t])
        
        
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
        mu_wc = np.zeros((self.T, self.nx, 1))
        sigma_wc = np.zeros((self.T, self.nx, self.nx))

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
        x_mean[0] = self.kalman_filter(self.v_mean_hat[0], self.M_hat[0], self.x0_mean_hat, self.x_cov[0], y[0]) #initial state estimation

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
            x_mean[t+1] = self.kalman_filter(self.v_mean_hat[t+1], self.M_hat[t+1], x_mean[t], self.x_cov[t+1], y[t+1], mu_wc[t], u=u[t])
        
        self.J_mse = np.zeros(self.T + 1) # State estimation error MSE
        
            
        #Collect Estimation MSE 
        for t in range(self.T):
            if t==0:
                self.J_mse[t] = (x_mean[t]-x[t]).T@(self.S[0])@(x_mean[t]-x[t])
            else:
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
        sigma_wc = np.zeros((self.T, self.nx, self.nx))

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
        x_mean[0] = self.kalman_filter(self.v_mean_hat[0], self.M_hat[0], self.x0_mean_hat, self.x_cov[0], y[0]) #initial state estimation

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
            x_mean[t+1] = self.kalman_filter(self.v_mean_hat[t+1], self.M_hat[t+1], x_mean[t], self.x_cov[t+1], y[t+1], mu_wc[t], u=u[t])
        
        self.J_mse = np.zeros(self.T + 1) # State estimation error MSE
        
            
        #Collect Estimation MSE 
        for t in range(self.T):
            if t==0:
                self.J_mse[t] = (x_mean[t]-x[t]).T@(self.S[0])@(x_mean[t]-x[t])
            else:
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
