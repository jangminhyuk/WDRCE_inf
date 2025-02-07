#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
import cvxpy as cp
import scipy

# Distributionally Robust Control with DRMMSE
class DRCMMSE:
    def __init__(self, lambda_, theta_w, theta_v, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat, use_lambda, use_optimal_lambda):
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
        self.previousM = self.M_hat
        self.previousX = self.x0_cov_hat
        self.theta_w = theta_w
        self.theta_v = theta_v
        self.theta_x0 = theta_x0
        self.theta_x = theta_x0 # Only for DRCMMSE !!!!!!
        self.lambda_ = lambda_
        
        self.DR_sdp = self.create_DR_sdp()
        self.DR_sdp_init = self.create_DR_sdp_initial()
        print("------------------")
        print("DRCMMSE")
        if use_lambda==True or use_optimal_lambda==True: # Use given Lambda!!
            self.lambda_ = np.array([lambda_])
        else:
            self.lambda_ = self.optimize_penalty() #optimize penalty parameter for theta
            
        #print(self.lambda_)
        
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
        # Find inf_penalty (infimum value of penalty coefficient satisfying Assumption)
        self.infimum_penalty = self.binarysearch_infimum_penalty_finite()
        print("Infimum penalty:", self.infimum_penalty)
        print("Optimizing lambda . . . Please wait for a while")
        output = minimize(self.objective, x0=np.array([4*self.infimum_penalty]), method='L-BFGS-B', options={'eps': 1e-4 , 'disp': False, 'maxiter': 2000})

        optimal_penalty = output.x
        print("DRCMMSE Optimal penalty (lambda_star):", optimal_penalty[0])
        print(optimal_penalty)
        return optimal_penalty

    def objective(self, penalty):
        
        #Compute the upper bound in Proposition 1
        P = np.zeros((self.T+1, self.nx,self.nx))        
        S = np.zeros((self.T+1, self.nx,self.nx))
        r = np.zeros((self.T+1, self.nx,1))
        z = np.zeros((self.T+1, 1))
        z_tilde = np.zeros((self.T+1, 1))

        if np.max(np.linalg.eigvals(P)) > penalty:
        #or np.max(np.linalg.eigvals(P + S)) > penalty:
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
        
        x_cov = np.zeros((self.T+1, self.nx, self.nx))
        S_opt = np.zeros((self.T+1, self.nx + self.ny, self.nx + self.ny))
        S_xx = np.zeros((self.T+1, self.nx, self.nx))
        S_xy = np.zeros((self.T+1, self.nx, self.ny))
        S_yy = np.zeros((self.T+1, self.ny, self.ny))
        sigma_wc = np.zeros((self.T, self.nx, self.nx))
        
        x_cov[0], S_xx[0], S_xy[0], S_yy[0], _= self.DR_kalman_filter_cov_initial(self.M_hat[0], self.x0_cov_hat)
        for t in range(0, self.T-1):
            x_cov[t+1], S_xx[t+1], S_xy[t+1], S_yy[t+1], sigma_wc[t], z_tilde[t] = self.DR_kalman_filter_cov(P[t+1], S[t+1], self.M_hat[t+1], x_cov[t], self.Sigma_hat[t], penalty)

        y = self.get_obs(self.x0_init, self.true_v_init)
        x0_mean = self.DR_kalman_filter(self.v_mean_hat[0], self.M_hat[0], self.x0_mean_hat, y, S_xx[0], S_xy[0], S_yy[0]) #initial state estimation
        obj_val = penalty*self.T*self.theta_w**2 + (self.x0_mean_hat.T @ P[0] @ self.x0_mean_hat)[0][0] + 2*(r[0].T @ self.x0_mean_hat)[0][0] + z[0][0] + np.trace(P[0] @ S_xx[0]) + np.trace(S[0] @ x_cov[0]) + z_tilde.sum()

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
    
    
    def create_DR_sdp(self):
        V = cp.Variable((self.nx, self.nx), symmetric=True, name='V')
        Sigma_wc = cp.Variable((self.nx, self.nx), symmetric=True, name='Sigma_wc')
        Y = cp.Variable((self.nx,self.nx), name='Y')
        X_hat = cp.Variable((self.nx,self.nx), symmetric=True, name='X_hat')
        X_pred = cp.Variable((self.nx,self.nx), symmetric=True, name='X_pred')
        M_test = cp.Variable((self.ny, self.ny), symmetric=True, name='M_test')
        Z = cp.Variable((self.ny, self.ny), name='Z')
        L = cp.Variable((self.nx,self.nx), name='L')
        
        #Parameters
        S_var = cp.Parameter((self.nx,self.nx), name='S_var')
        P_var = cp.Parameter((self.nx,self.nx), name='P_var')
        Lambda_ = cp.Parameter(1, name='Lambda_')
        Sigma_w = cp.Parameter((self.nx, self.nx), name='Sigma_w') # nominal Sigma_w
        x_cov = cp.Parameter((self.nx, self.nx), name='x_cov') # x_cov from before time step
        M_hat = cp.Parameter((self.ny, self.ny), name='M_hat')
        radi = cp.Parameter(nonneg=True, name='radi')
        radi_x = cp.Parameter(nonneg=True, name='radi_x')
        
        #use Schur Complements
        #obj function
        obj = cp.Maximize(cp.trace(S_var @ V) + cp.trace((P_var - Lambda_ * np.eye(self.nx)) @ Sigma_wc) + 2*Lambda_*cp.trace(Y)) 
        
        #constraints
        constraints = [
                cp.bmat([[Sigma_w, Y],
                         [Y.T, Sigma_wc]
                         ]) >> 0,
                X_hat == self.A @ x_cov @ self.A.T + Sigma_wc,
                cp.bmat([[X_pred-V, X_pred @ self.C.T],
                        [self.C @ X_pred, self.C @ X_pred @ self.C.T + M_test]
                        ]) >> 0 ,
                self.C @ X_pred @ self.C.T + M_test >>0,
                cp.trace(M_hat + M_test - 2*Z ) <= radi**2,
                cp.trace(X_hat + X_pred - 2*L ) <= radi_x**2,
                cp.bmat([[X_hat, L],
                         [L.T, X_pred]
                         ]) >> 0,
                cp.bmat([[M_hat, Z],
                         [Z.T, M_test]
                         ]) >> 0,
                X_hat >>0,
                V>>0,
                X_pred >>0,
                M_test >>0,
                Sigma_wc >>0
                ]
        
        prob = cp.Problem(obj, constraints)
        return prob
        
    def solve_DR_sdp(self, prob, P_t1, S_t1, M, X_cov, Sigma_hat, theta, Lambda_):
        #construct problem
        params = prob.parameters()
        params[0].value = S_t1 # S[t+1]
        params[1].value = P_t1 # P[t+1]
        params[2].value = Lambda_
        params[3].value = Sigma_hat
        params[4].value = X_cov
        params[5].value = M # Noise covariance
        params[6].value = theta
        params[7].value = self.theta_x
        
        
        prob.solve(solver=cp.MOSEK)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'False in DRCMMSE combined!!!!!!!!!!!!!')
        
        sol = prob.variables()
        #['V', 'Sigma_wc', 'Y', 'X_hat', 'X_pred', 'M_test', 'Z', 'L']
        
        S_xx_opt = sol[4].value
        S_xy_opt = S_xx_opt @ self.C.T
        S_yy_opt = self.C @ S_xx_opt @ self.C.T + sol[5].value
        
        Sigma_wc_opt = sol[1].value
        cost = prob.value
        return  S_xx_opt, S_xy_opt, S_yy_opt, Sigma_wc_opt, cost
    
    
    def create_DR_sdp_initial(self):
        #construct problem
        #Variables
        X0 = cp.Variable((self.nx, self.nx), symmetric=True, name='X0') # prediction
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        M0 = cp.Variable((self.ny, self.ny), symmetric=True, name='M0')
        Z = cp.Variable((self.ny, self.ny), name='Z')
        Y = cp.Variable((self.nx, self.nx), name='Y')
        
        S_var = cp.Parameter((self.nx,self.nx), name='S_var')
        M0_hat = cp.Parameter((self.ny, self.ny), name='M0_hat') # nominal noise covariance
        theta_v0 = cp.Parameter(nonneg=True, name='theta_v0')
        X0_hat = cp.Parameter((self.nx, self.nx), name='X0_hat')
        theta_x0 = cp.Parameter(nonneg=True, name='theta_x0')

        #use Schur Complements
        #obj function
        obj = cp.Maximize(cp.trace(S_var @ X)) 
        
        #constraints
        constraints = [
                cp.bmat([[X0-X, X0 @ self.C.T],
                        [self.C @ X0, self.C @ X0 @ self.C.T + M0]
                        ]) >> 0 ,
                cp.trace(M0_hat + M0 - 2*Z ) <= theta_v0**2,
                cp.bmat([[M0_hat, Z],
                         [Z.T, M0]
                         ]) >> 0,
                cp.trace(X0_hat + X0 - 2*Y ) <= theta_x0**2,
                cp.bmat([[X0_hat, Y],
                         [Y.T, X0]
                         ]) >> 0,
                X>>0,
                X0>>0,
                M0>>0
                ]
        
        prob = cp.Problem(obj, constraints)


        return prob
    

    
    def solve_DR_sdp_initial(self, prob, M_hat, X_hat):
        params = prob.parameters()
        params[0].value = self.S[0]
        params[1].value = M_hat
        params[2].value = self.theta_v
        params[3].value = X_hat
        params[4].value = self.theta_x0
        
        
        prob.solve(solver=cp.MOSEK)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'False in DRCMMSE initial!!!!!!!!!!!!!')
        
        sol = prob.variables()
        S_xx_opt = sol[0].value
        S_xy_opt = S_xx_opt @ self.C.T
        S_yy_opt = self.C @ S_xx_opt @ self.C.T + sol[2].value
        cost = prob.value
        return  S_xx_opt, S_xy_opt, S_yy_opt, cost
    
    #DR Kalman FILTER !
    def DR_kalman_filter(self, v_mean_hat, M_hat, x, y, S_xx, S_xy, S_yy, mu_w=None, u = None):
        if u is None:
            #Initial state estimate
            x_ = x
            y_ = self.C @ x 
        else:
            #Prediction step
            x_ = self.A @ x + self.B @ u + mu_w
            y_ = self.C @ (self.A @ x + self.B @ u + mu_w) + v_mean_hat
        
        x_new = S_xy @ np.linalg.inv(S_yy) @ (y - y_) + x_
        return x_new
    
    def DR_kalman_filter_cov(self, P, S, M_hat, X_cov, Sigma_hat, Lambda):
        #Performs state estimation based on the current state estimate, control input and new observation
        theta = self.theta_v
        S_xx, S_xy, S_yy, Sigma_wc, cost = self.solve_DR_sdp(self.DR_sdp, P, S, M_hat, X_cov, Sigma_hat, theta, Lambda)
        
        X_cov_new = S_xx - S_xy @ np.linalg.inv(S_yy) @ S_xy.T
       
        return X_cov_new, S_xx, S_xy, S_yy, Sigma_wc, cost
    
    def DR_kalman_filter_cov_initial(self, M_hat, X_cov): #DRKF !!
        #Performs state estimation based on the current state estimate, control input and new observation
        S_xx, S_xy, S_yy, cost = self.solve_DR_sdp_initial(self.DR_sdp_init, M_hat, X_cov)
        
        X_cov_new = S_xx - S_xy @ np.linalg.inv(S_yy) @ S_xy.T
        return X_cov_new, S_xx, S_xy, S_yy, cost
    
    def riccati(self, Phi, P, S, r, z, Sigma_hat, mu_hat, lambda_, t):
        #Riccati equation corresponding to the Theorem 1

        temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
        P_ = self.Q + self.A.T @ temp @ P @ self.A
        S_ = self.Q + self.A.T @ P @ self.A - P_

        # Double check Assumption 1
        if lambda_ <= np.max(np.linalg.eigvals(P)):
        #or lambda_ <= np.max(np.linalg.eigvals(P+S)):
            print("t={}: lambda check False!!!!!!!!!".format(t))
            print("np.max(np.linalg.eigvals(P) : ", np.max(np.linalg.eigvals(P)))
            return None
        r_ = self.A.T @ temp @ (r + P @ mu_hat)
        z_ = z + - lambda_* np.trace(Sigma_hat) \
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
        
        self.x_cov[0], self.S_xx[0], self.S_xy[0], self.S_yy[0], _= self.DR_kalman_filter_cov_initial(self.M_hat[0], self.x0_cov_hat)
        for t in range(self.T):
            print("DRCMMSE Offline step : ",t,"/",self.T)
            self.x_cov[t+1], self.S_xx[t+1], self.S_xy[t+1], self.S_yy[t+1], self.sigma_wc[t], _ = self.DR_kalman_filter_cov(self.P[t+1], self.S[t+1], self.M_hat[t+1], self.x_cov[t], self.Sigma_hat[t], self.lambda_)
            

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
                'mse':self.J_mse}


