#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
import cvxpy as cp
import scipy
import control
from joblib import Parallel, delayed

class inf_DRCE:
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
        self.x0_mean = x0_mean 
        self.x0_cov = x0_cov
        self.x0_mean_hat = x0_mean_hat
        self.x0_cov_hat = x0_cov_hat
        self.mu_hat = mu_hat
        self.Sigma_hat = Sigma_hat # Sigma_w_hat
        self.mu_w = mu_w
        self.mu_v = mu_v
        self.Sigma_w = Sigma_w
        self.theta_w = theta_w
        self.theta_v = theta_v
        self.theta_x0 = theta_x0
        self.error_bound = 1e-5
        self.max_iteration = 1000
        
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
            
        print("inf DRCE ", self.dist, " / ", self.noise_dist)    
        self.theta_w = theta_w
        self.theta_v = theta_v
        self.theta_x0 = theta_x0
        self.lambda_ = lambda_
        
        
        self.sdp_prob = self.create_DR_sdp()

        if use_lambda==True or use_optimal_lambda==True:
            self.lambda_ = np.array([lambda_])
        else:
            self.lambda_ = self.optimize_penalty() #optimize penalty parameter for theta
            #print(self.lambda_)
        self.Phi = self.B @ np.linalg.inv(self.R) @ self.B.T - 1/self.lambda_ * np.eye(self.nx)
        self.flag= True
        
    
    def optimize_penalty(self):
        # Find inf_penalty (infimum value of penalty coefficient satisfying Assumption 1)
        self.infimum_penalty = self.binarysearch_infimum_penalty_finite()
        print("inf DRCE Infimum penalty:", self.infimum_penalty)
        
        # Optimize penalty
        print("Optimizing lambda . . . Please wait")
        output = minimize(self.objective, x0=np.array([4*self.infimum_penalty]), method='L-BFGS-B', options={'disp': False, 'maxiter': 1000,'ftol': 1e-6,'gtol': 1e-6, 'maxfun':1000})
        optimal_penalty = output.x
        print("infDRCE Optimal penalty (lambda_star) :", optimal_penalty[0], " when theta_w : ", self.theta_w, "\n\n")
        return optimal_penalty
        
        # output = minimize(self.objective, x0=np.array([4*self.infimum_penalty]), method='L-BFGS-B', options={'disp': False, 'maxiter': 100,'ftol': 1e-6,'gtol': 1e-6, 'maxfun':100})
        # optimal_penalty = output.x
        # print("inf WDR-CE Optimal penalty (lambda_star) :", optimal_penalty[0], " when theta_w : ", self.theta_w, "\n\n")
        # return optimal_penalty

        #### BELOW 
        # penalty_values = np.linspace(3* self.infimum_penalty, 10 * self.infimum_penalty, num=10)
        
        # # Uncomment below for parallel computation
        # # objectives = Parallel(n_jobs=-1)(delayed(self.objective)(np.array([p])) for p in penalty_values)
        
        # # ----
        # objectives = []

        # # Compute objectives sequentially
        # for p in penalty_values:
        #     obj = self.objective(np.array([p]))
        #     objectives.append(obj)
        # # ----
        
        # objectives = np.array(objectives)
        # optimal_penalty = penalty_values[np.argmin(objectives)]
        # #optimal_penalty = output.x
        # print("infDRCE Optimal penalty (lambda_star):", optimal_penalty, "theta_w : ", self.theta_w, " theta_v : ", self.theta_v)
        # return np.array([optimal_penalty])
    
    def objective(self, penalty):
        #Compute the upper bound in Proposition 1
        P = np.zeros((self.nx,self.nx))        
        S = np.zeros((self.nx,self.nx))
        r = np.zeros((self.nx,1))
        z = 0

        if np.max(np.linalg.eigvals(P)) > penalty:
                return np.inf

        for t in range(0, self.max_iteration):
            Phi = self.B @ np.linalg.inv(self.R) @ self.B.T - 1/penalty * np.eye(self.nx)
            P_temp, S_temp, r_temp, z_temp, K_temp, L_temp, H_temp, h_temp, g_temp = self.riccati(Phi, P, S, r, z, self.Sigma_hat, self.mu_hat, penalty, t)
            if np.max(np.linalg.eigvals(P_temp)) > penalty or np.max(np.linalg.eigvals(P_temp + S_temp)) > penalty:
                return np.inf

            max_diff = 0
            for row in range(len(P)):
                for col in range(len(P)):
                    if abs(P[row, col] - P_temp[row, col]) > max_diff:
                        max_diff = abs(P[row, col] - P_temp[row, col])
            P = P_temp
            S = S_temp
            r = r_temp
            if max_diff < self.error_bound:
                P_ss = P
                S_ss = S
                r_ss = r
                temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
#                sdp_prob = self.gen_sdp(penalty)
                P_post_ss, sigma_wc_ss, M_wc, z_tilde_ss, status = self.KF_riccati(self.x0_cov_hat, P_ss, S_ss, penalty)
                if status in ["infeasible", "unbounded", "unknown"]:
                    print(status)
                    return np.inf
                if np.max(sigma_wc_ss) >= 1e2:
                    return np.inf
                rho = (2*self.mu_hat - Phi @ r_ss).T @ temp @ r_ss - penalty* np.trace(self.Sigma_hat) + self.mu_hat.T @ temp @ P_ss @ self.mu_hat + z_tilde_ss
#                print('Lambda: ', penalty, 'Objective: ', penalty*self.theta**2 + rho[0])
                return penalty*self.theta_w**2 + rho[0]
    
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
        P = np.zeros((self.nx,self.nx))
        S = np.zeros((self.nx,self.nx))
        r = np.zeros((self.nx,1))
        z = np.zeros((1,1))
        if penalty < 0:
            return False
        if np.max(np.linalg.eigvals(P+S)) >= penalty:
        #or np.max(np.linalg.eigvals(P + S)) >= penalty:
                return False
        for t in range(0, self.max_iteration):
            Phi = self.B @ np.linalg.inv(self.R) @ self.B.T - 1/penalty * np.eye(self.nx)
            P_temp, S_temp, r_temp, z_temp, K_temp, L_temp, H_temp, h_temp, g_temp = self.riccati(Phi, P, S, r, z, self.Sigma_hat, self.mu_hat, penalty, t)
            if np.max(np.linalg.eigvals(P_temp+S_temp)) > penalty or np.max(np.linalg.eigvals(P_temp)) > penalty:
                return False
            max_diff = 0
            for row in range(len(P)):
                for col in range(len(P[0])):
                    if abs(P[row, col] - P_temp[row, col]) > max_diff:
                        max_diff = abs(P[row, col] - P_temp[row, col])
            P = P_temp
            S = S_temp
            r = r_temp
            z = z_temp
            if max_diff < self.error_bound:
                P_post_ss, sigma_wc_ss, M_wc, z_tilde_ss, status = self.KF_riccati(self.x0_cov_hat, P, S, np.array([penalty]))
                if status in ["infeasible", "unbounded"]:
                    #print(status)
                    return False
                if np.max(sigma_wc_ss) >= 1e2:
                    return False
                return True
        print("Minimax Riccati iteration did not converge : inf DRCE")

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
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        Sigma_wc = cp.Variable((self.nx, self.nx), symmetric=True, name='Sigma_wc')
        Y = cp.Variable((self.nx,self.nx), name='Y')
        X_pred = cp.Variable((self.nx,self.nx), symmetric=True, name='X_pred')
        M_wc = cp.Variable((self.ny, self.ny), symmetric=True, name='M_wc')
        Z = cp.Variable((self.ny, self.ny), name='Z')
        
        #Parameters
        S_var = cp.Parameter((self.nx,self.nx), name='S_var')
        P_var = cp.Parameter((self.nx,self.nx), name='P_var')
        Lambda_ = cp.Parameter(1, name='Lambda_')
        Sigma_w_hat = cp.Parameter((self.nx, self.nx), name='Sigma_w_hat') # nominal Sigma_w
        M_hat = cp.Parameter((self.ny, self.ny), name='M_hat')
        radi = cp.Parameter(nonneg=True, name='radi')
               
        #use Schur Complements
        #obj function
        obj = cp.Maximize(cp.trace(S_var @ X) + cp.trace((P_var - Lambda_ * np.eye(self.nx)) @ Sigma_wc) + 2*Lambda_*cp.trace(Y)) 
        
        #constraints
        constraints = [
                cp.bmat([[Sigma_w_hat, Y],
                         [Y.T, Sigma_wc]
                         ]) >> 0,
                X_pred == self.A @ X @ self.A.T + Sigma_wc,
                cp.bmat([[X_pred-X, X_pred @ self.C.T],
                        [self.C @ X_pred, self.C @ X_pred @ self.C.T + M_wc]
                        ]) >> 0 ,
                self.C @ X_pred @ self.C.T + M_wc >>0,
                cp.trace(M_hat + M_wc - 2*Z ) <= radi**2,
                cp.bmat([[M_hat, Z],
                         [Z.T, M_wc]
                         ]) >> 0,
                X>>0,
                X_pred >>0,
                M_wc >>0,
                Sigma_wc >>0
                ]
        
        prob = cp.Problem(obj, constraints)
        return prob
        
    
    def solve_DR_sdp(self, prob, Lambda_, P_ss, S_ss, Sigma_hat, M_hat):
        #construct problem
        params = prob.parameters()
        params[0].value = S_ss
        params[1].value = P_ss
        params[2].value = Lambda_
        params[3].value = Sigma_hat
        params[4].value = M_hat# Noise covariance
        params[5].value = self.theta_v
        
        
        prob.solve(solver=cp.MOSEK)
        
        # if prob.status in ["infeasible", "unbounded"]:
        #     print(prob.status, 'False in inf DRCE !!!!!!!!!!!!!')
        #     print("Current Lambda: ", Lambda_)
        
        sol = prob.variables()
        #['X', 'Sigma_wc', 'Y', 'X_pred', 'M_wc', 'Z']
        
        # self.S_xx_opt = sol[3].value
        # self.S_xy_opt = self.S_xx_opt @ self.C.T
        # self.S_yy_opt = self.C @ self.S_xx_opt @ self.C.T + sol[4].value
        M_wc_opt = sol[4].value 
        #S_opt = S.value
        Sigma_wc_opt = sol[1].value
        cost = prob.value
        status = prob.status
        return  Sigma_wc_opt, M_wc_opt, sol[0].value, cost, status
    
    def kalman_filter(self, v_mean_hat, x, y, mu_w, u = None):
        #Performs state estimation based on the current state estimate, control input and new observation
        if u is None:
            #Initial state estimate
            x_ = x
        else:
            #Prediction update
            x_ = self.A @ x + self.B @ u + mu_w

        #Measurement update
        resid = y - (self.C @ x_ + v_mean_hat) 
        temp = np.linalg.solve(self.M_wc, resid)
        x_new = x_ + self.X_pred @ self.C.T @ temp
        return x_new
    
    def KF_riccati(self, X_pred, P_ss, S_ss, lambda_):
        sdp_prob = self.create_DR_sdp()
        sigma_wc_1, M_wc, X_post_ ,z_tilde__, stat_ = self.solve_DR_sdp(sdp_prob, lambda_,  P_ss, S_ss, self.Sigma_hat, self.M_hat)
        
        # X_pred_ = X_pred
        # for t in range(self.max_iteration):
        #     X_pred_temp_ = self.A @ (X_pred_ - X_pred_ @ self.C.T @ np.linalg.solve(self.C @ X_pred_ @ self.C.T + M_wc, self.C @ X_pred_)) @ self.A.T + sigma_wc_1
        #     if min(np.linalg.eigvals(self.C @ X_pred_temp_ @ self.C.T + M_wc)) <= 0:
        #         print("error in KF_riccati !!!")    
        #         return np.inf
    
        #     max_diff = 0
        #     for row in range(len(X_pred_)):
        #        for col in range(len(X_pred_[0])):
        #            if abs(X_pred_[row, col] - X_pred_temp_[row, col]) > max_diff:
        #                max_diff = X_pred_[row, col] - X_pred_temp_[row, col]
        #     X_pred_ = X_pred_temp_
        #     X_post_ = X_pred_ - X_pred_ @ self.C.T @ np.linalg.solve(self.C @ X_pred_ @ self.C.T + M_wc, self.C @ X_pred_)
            
        #     #print("X_post_", np.linalg.norm(X_post_), "/ max_diff : ", max_diff)
        #     if abs(max_diff) < self.error_bound:
        #         if np.linalg.matrix_rank(control.ctrb(self.A, scipy.linalg.sqrtm(sigma_wc_1))) < self.nx:
        #                 print('(A, sqrt(Sigma)) is not controllable!!!!!!')
        #                 stat_ = "infeasible"
        #         return X_post_, sigma_wc_1, M_wc, z_tilde__, stat_
        
        return X_post_, sigma_wc_1, M_wc, z_tilde__, stat_

    def riccati(self, Phi, P, S, r, z, Sigma_hat, mu_hat, lambda_, t):
        #Riccati equation corresponding to the Theorem 1

        temp = np.linalg.inv(np.eye(self.nx) + P @ Phi)
        P_ = self.Q + self.A.T @ temp @ P @ self.A
        S_ = self.Q + self.A.T @ P @ self.A - P_

        # Double check Assumption 1
        if lambda_ <= np.max(np.linalg.eigvals(P)):
        #or lambda_ <= np.max(np.linalg.eigvals(P+S)):
            print("t={}: False!!!!!!!!!".format(t))
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
        #Compute P, S, r, z, K and L, as well as the worst-case distribution parameters H, h and g backward in time
        #\bar{w}_t^* = H[t] \bar{x}_t + h[t], \Sigma_t^* = g[t]
        offline_start = time.time()
        self.P_list = np.zeros((self.max_iteration+1, self.nx, self.nx))
        self.S_list = np.zeros((self.max_iteration+1, self.nx, self.nx))
        P = np.zeros((self.nx, self.nx))
        S = np.zeros((self.nx, self.nx))
        self.P_list[0,:,:] = P
        self.S_list[0,:,:] = S
        
        r = np.zeros((self.nx, 1))
        z = 0

        for t in range(self.max_iteration):
            P_temp, S_temp, r_temp, z_temp, K_temp, L_temp, H_temp, h_temp, g_temp = self.riccati(self.Phi, P, S, r, z, self.Sigma_hat, self.mu_hat, self.lambda_, t)
            max_diff = 0
            for row in range(len(P)):
                for col in range(len(P[0])):
                    if abs(P[row, col] - P_temp[row, col]) > max_diff:
                        max_diff = abs(P[row, col] - P_temp[row, col])
            P = P_temp
            S = S_temp
            self.P_list[t+1,:,:] = P
            self.S_list[t+1,:,:] = S
            r = r_temp
            z = z_temp
            if max_diff < self.error_bound:
                self.P_ss = P
                self.S_ss = S
                self.r_ss = r
                temp2 = np.linalg.solve(self.R, self.B.T)
                temp = np.linalg.inv(np.eye(self.nx) + P @ self.Phi)
                self.K_ss = - temp2 @ temp @ P @ self.A
                self.L_ss = - temp2 @ temp @ (r + P @ self.mu_hat)
                self.h_ss = np.linalg.inv(self.lambda_ * np.eye(self.nx) - P) @ (r + P @ self.B @ self.L_ss + self.lambda_*self.mu_hat)
                self.H_ss = np.linalg.inv(self.lambda_* np.eye(self.nx)  - P) @ P @ (self.A + self.B @ self.K_ss)
                self.g_ss = self.lambda_**2 * np.linalg.inv(self.lambda_*np.eye(self.nx) - P - S) @ self.Sigma_hat @ np.linalg.inv(self.lambda_*np.eye(self.nx) - P - S)
                self.max_it_P = t
                self.P_list[t+1:,:,:] = P
                self.S_list[t+1:,:,:] = S
                P_post, sigma_wc, M_wc, z_tilde_, status = self.KF_riccati(self.x0_cov_hat, self.P_ss, self.S_ss, self.lambda_)
                
                #
                
                #
                self.X_pred = P_post
                self.sigma_wc = sigma_wc
                self.M_wc = M_wc
                self.z_tilde = z_tilde_
                
                offline_end = time.time()
                self.offline_time = offline_end - offline_start
                return
            
        print("Minimax Riccati iteration did not converge : inf DRCE")
        self.P_ss = P
        self.S_ss = S
        self.r_ss = r
        self.K_ss = None
        self.L_ss = None
        self.h_ss = None
        self.H_ss = None
        self.g_ss = None
        self.flag = False
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
        x_mean[0] = self.kalman_filter(self.v_mean_hat, self.x0_mean_hat, y[0], self.mu_hat) #initial state estimation

        for t in range(self.T):
            mu_wc[t] = self.H_ss @ x_mean[t] + self.h_ss #worst-case mean
            
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
            u[t] = self.K_ss @ x_mean[t] + self.L_ss
            x[t+1] = self.A @ x[t] + self.B @ u[t] + true_w
            y[t+1] = self.get_obs(x[t+1], true_v)

            #Update the state estimation (using the worst-case mean and covariance)
            x_mean[t+1] = self.kalman_filter(self.v_mean_hat, x_mean[t],  y[t+1], mu_wc[t], u=u[t])

        #State estimation error MSE
        self.J_mse = np.zeros(self.T + 1) 
        #Collect Estimation MSE 
        for t in range(self.T):
            self.J_mse[t] = (x_mean[t]-x[t]).T@(self.S_ss)@(x_mean[t]-x[t])

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


