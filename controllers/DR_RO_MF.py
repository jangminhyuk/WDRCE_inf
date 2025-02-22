#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from scipy.optimize import minimize
import cvxpy as cp
import scipy
from joblib import Parallel, delayed
from scipy.linalg import cholesky, solve_triangular

# Distributionally Robust regret-optimal with measurement feedback (DR-RO-MF)
class DR_RO_MF:
    def __init__(self, theta, N, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat, x0_cov_hat):
        self.dist = dist
        self.noise_dist = noise_dist
        self.N = N
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
        
        self.radius = theta # size of the ambiguity set
        
        self.F, self.G, self.J, self.L = self.compute_operator_matrices()
        self.S = np.eye(N*self.nu) + self.F @ self.F.T
        self.T = np.eye(N*self.nu) + self.F.T @ self.F
        self.V = np.eye(N*self.nx) + self.L.T @ self.L
        self.U = np.eye(N*self.nx) + self.L @ self.L.T
        self.E0 = - np.linalg.inv(self.T) @ self.F.T @ self.G @ self.L.T @ np.linalg.inv(self.U)
        
        
        # For S and U: factorize as S = S^{1/2}(S^{1/2})^T and U = U^{1/2}(U^{1/2})^T.
        S_half = cholesky(self.S, lower=True)  # S^{1/2} (lower triangular)
        self.U_half = cholesky(self.U, lower=True)  # U^{1/2} (lower triangular)

        # For T and V: factorize as T = (T^{1/2})^T T^{1/2} and V = (V^{1/2})^T V^{1/2}.
        # Here we set T^{1/2} and V^{1/2} to be lower triangular.
        self.T_half = cholesky(self.T, lower=True)  # T^{1/2} (lower triangular)
        self.V_half = cholesky(self.V, lower=True)  # V^{1/2} (lower triangular)
        # Then, by notation, T^{\top/2} = T_half.T and V^{\top/2} = V_half.T.

        # Compute the inverses of the square-root factors.
        # For T^{-1/2} and U^{-1/2} (and similarly for V), we compute the inverse of the Cholesky factor.
        T_half_inv = solve_triangular(self.T_half, np.eye(self.T_half.shape[0]), lower=True)
        U_half_inv = solve_triangular(self.U_half, np.eye(self.U_half.shape[0]), lower=True)
        V_half_inv = solve_triangular(self.V_half, np.eye(self.V_half.shape[0]), lower=True)
        # Then, (T^{-1/2})^T = (T_half_inv).T, etc.

        # Now implement the equations.

        # Equation for W:
        # W = -(T^{-1/2})^T F^T G L^T (U^{-1/2})^T
        self.W = - (T_half_inv.T) @ self.F.T @ self.G @ self.L.T @ (U_half_inv.T)

        # Equation for P:
        # P = (V^{-1/2})^T G^T F T^{-1/2}
        self.P = (V_half_inv.T) @ self.G.T @ self.F @ T_half_inv

        # Equation for Z:
        # Z = T^{1/2} E U^{1/2} - W
        #self.Z = self.T_half @ self.E0 @ self.U_half - self.W
        
        
        #self.DR_sdp = self.create_DR_sdp()
        
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
    import numpy as np

    def compute_operator_matrices(self):
        """
        Computes the block operator matrices F, G, J, and L for a discrete-time LTI system
        with dynamics
            x_{t+1} = A x_t + B u_t + w_t,
            y_t     = C x_t + v_t,
        where the operators are defined over a horizon T.
        
        The operators are strictly causal (strictly lower-triangular block Toeplitz):
            F: maps u to x, where F[t, j] = A^(t-j-1) B for t > j, 0 otherwise;
            G: maps w to x, where G[t, j] = A^(t-j-1) for t > j, 0 otherwise;
            J: maps u to y, where J[t, j] = C A^(t-j-1) B for t > j, 0 otherwise;
            L: maps w to y, where L[t, j] = C A^(t-j-1) for t > j, 0 otherwise.
        
        Parameters:
            T (int): the time horizon (number of time steps)
        
        Returns:
            F (ndarray): Block matrix of shape (T*n_x, T*n_u)
            G (ndarray): Block matrix of shape (T*n_x, T*n_x)
            J (ndarray): Block matrix of shape (T*n_y, T*n_u)
            L (ndarray): Block matrix of shape (T*n_y, T*n_x)
        """
        A = self.A  # shape: (n_x, n_x)
        B = self.B  # shape: (n_x, n_u)
        C = self.C  # shape: (n_y, n_x)
        
        n_x = A.shape[0]
        n_u = B.shape[1]
        n_y = C.shape[0]
        
        # horizon
        N = self.N
        
        # Initialize block matrices with zeros.
        F = np.zeros((N * n_x, N * n_u))
        G = np.zeros((N * n_x, N * n_x))
        J = np.zeros((N * n_y, N * n_u))
        L = np.zeros((N * n_y, N * n_x))
        
        
        
        # Loop over the time steps. Note that for t=0, there is no contribution
        # from u or w (assuming x0 is given/zero). So we start at t=1.
        for t in range(1, N):
            for j in range(t):
                # Compute A^(t-j-1)
                A_power = np.linalg.matrix_power(A, t - j - 1)
                
                # Fill in the (t,j) block for F: x_t += A^(t-j-1) B u_j
                F[t*n_x:(t+1)*n_x, j*n_u:(j+1)*n_u] = A_power @ B
                
                # Fill in the (t,j) block for G: x_t += A^(t-j-1) w_j
                G[t*n_x:(t+1)*n_x, j*n_x:(j+1)*n_x] = A_power
                
                # Fill in the (t,j) block for J: y_t = C x_t, so contribution from u_j is C A^(t-j-1)B
                J[t*n_y:(t+1)*n_y, j*n_u:(j+1)*n_u] = C @ A_power @ B
                
                # Fill in the (t,j) block for L: contribution from w_j is C A^(t-j-1)
                L[t*n_y:(t+1)*n_y, j*n_x:(j+1)*n_x] = C @ A_power
                
        return F, G, J, L

    
    def create_DR_sdp(self):
        N = self.N
        
        # Primary decision variables.
        X = cp.Variable((N*(self.nx + self.ny), N*(self.nx + self.ny)), 
                        symmetric=True, name='X')
        Y = cp.Variable((N*self.nu, N*self.ny), name='Y')  
        # sub-blocks of X:
        X11 = cp.Variable((N*self.nx, N*self.nx), symmetric=True, name='X11')
        X12 = cp.Variable((N*self.nx, N*self.ny), name='X12')
        X22 = cp.Variable((N*self.ny, N*self.ny), symmetric=True, name='X22')
        
        
        star = cp.Variable((N*self.nu, N*self.nx), name='star')
        
        
        gamma = cp.Parameter(nonneg=True, name='gamma')
        
        
        I_nx = cp.Constant(np.eye(N * self.nx))
        I_nu = cp.Constant(np.eye(N * self.nu))
        I_ny = cp.Constant(np.eye(N * self.ny))
        
        Zero_xy = cp.Constant(np.zeros((N*self.nx, N*self.ny)))
        Zero_xu = cp.Constant(np.zeros((N*self.nx, N*self.nu)))
        Zero_yu = cp.Constant(np.zeros((N*self.ny, N*self.nu)))
        
        # Objective
        # min or max? Usually it's Minimize for an SDP, but your snippet had no "cp.Minimize(...)"
        # We'll assume Minimize.
        obj = cp.Minimize(
            gamma * (self.radius**2 - N*(self.nx + self.ny)) + cp.trace(X)
        )
        
        # Constraints
        constraints = []
        
        # 1) Y is lower triangular
        constraints.append(Y == cp.tril(Y))
        
        # 2) X is PSD
        constraints.append(X >> 0)
        
        # 3) X must equal the block composition of [X11, X12; X12^T, X22]
        constraints.append(
            X == cp.bmat([[X11,    X12],
                        [X12.T,  X22]])
        )
        
        # 4) star = M_gamma_inv @ Y + H_gamma
        #    Because star must equal that expression
        constraints.append(star == self.M_gamma_inv @ Y + self.H_gamma)
        
        # 5) Big LMI block #1
        # Make sure all blocks line up dimensionally!
        big_block_1 = cp.bmat([
            [X11,          X12,           gamma * I_nx,     Zero_xy,          Zero_xu],
            [X12.T,        X22,           Zero_xy.T,        gamma * I_ny,     Zero_yu],
            [gamma * I_nx, Zero_xy,       gamma * I_nx,     -self.P @ star,   Zero_xu],
            [Zero_xy.T,    gamma * I_ny,  -star.T @ self.P.T, gamma * I_ny,   star.T],
            [Zero_xu.T,    Zero_yu.T,     Zero_xu.T,        star,            I_nu]
        ])
        constraints.append(big_block_1 >> 0)
        
        # 6) Big LMI block #2
        #    This must also be PSD. Check shapes carefully!
        #    If Y and self.W_minus_gamma are each (N*self.nu, N*self.ny),
        #    then (Y - self.W_minus_gamma) is (N*self.nu, N*self.ny).
        #    So (Y - self.W_minus_gamma).T is (N*self.ny, N*self.nu).
        #
        #    For this block to be conformable with I_nx in the top-left, 
        #    you'd need Nx == Nu, etc. 
        #    Possibly you meant to use I_nu in the top-left, or I_ny, etc.
        #    We'll guess I_nu to match the row dimension of Y.
        
        big_block_2 = cp.bmat([
            [I_nu,                       (Y - self.W_minus_gamma).T],
            [Y - self.W_minus_gamma,     I_nu]
        ])
        constraints.append(big_block_2 >> 0)
        
        # Form the problem
        prob = cp.Problem(obj, constraints)
        return prob
        
    def solve_DR_sdp(self, prob, gamma):
        
        N = self.N
        
        #    M_gamma = (gamma^{-1} I + gamma^{-2} P^T P)^{1/2}.
        M_temp = (gamma**(-1)) * np.eye(N * self.nx) + (gamma**(-2)) * (self.P.T @ self.P)
        self.M_gamma = cholesky(M_temp, lower=True)  # M_gamma is lower triangular and positive definite.

        self.W_gamma = self.M_gamma @ self.W

        #    Here we define W_{+,gamma} as the lower-triangular part (including the diagonal).
        self.W_plus_gamma = np.tril(self.W_gamma, k=0)
        # compute the strictly anticausal part as:
        self.W_minus_gamma = self.W_gamma - self.W_plus_gamma  # strictly anticausal (i.e. the strictly upper triangular part)

        #    Y = M_gamma @ (T^{1/2} E U^{1/2}) - W_{+,gamma}.
        #    Here T^{1/2} is stored in T_half and U^{1/2} in U_half.
        #Y = self.M_gamma @ (self.T_half @ self.E0 @ self.U_half) - self.W_plus_gamma
        
        self.M_gamma_inv = solve_triangular(self.M_gamma, np.eye(self.M_gamma.shape[0]), lower=True)
        self.H_gamma = solve_triangular(self.M_gamma, self.W_plus_gamma, lower=True) - self.W
        
        
        self.DR_sdp = self.create_DR_sdp()
        
        #construct problem
        params = prob.parameters()
        # for i, param in enumerate(params):
        #     print(f"params[{i}]: {param.name()}")
        params[0].value = gamma
        
        
        prob.solve(solver=cp.MOSEK)
        if prob.status in ["infeasible", "unbounded"]:
            print("radius : ", self.radius)
            print("gamma : ", gamma)
            print(prob.status, 'False in DR RO MF SDP !!!!!!!!')
        
        sol = prob.variables()
        # for i, var in enumerate(sol):
        #     print(f"var[{i}]: {var.name()}")
        
        
        cost = prob.value
        return   cost, prob.status
    
    

    def get_obs(self, x, v):
        #Get new noisy observation
        obs = self.C @ x + v
        return obs

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

