import numpy as np
import cvxpy as cp
import mosek
import control

np.random.seed(42)

def uniform(a, b, N=1):
    n = a.shape[0]
    return a[:, None] + (b[:, None] - a[:, None]) * np.random.rand(n, N)

def normal(mu, Sigma, N=1):
    return np.random.multivariate_normal(mu.ravel(), Sigma, size=N).T

def quad_inverse(x, b, a):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            beta = 0.5 * (a[j] + b[j])
            alpha = 12.0 / ((b[j] - a[j]) ** 3)
            tmp = 3 * x[i, j] / alpha - (beta - a[j]) ** 3
            x[i, j] = beta + (tmp if tmp >= 0 else -(-tmp) ** (1. / 3.)) ** (1. / 3.)
    return x

def quadratic(wmax, wmin, N=1):
    x = np.random.rand(N, wmin.shape[0]).T
    return quad_inverse(x, wmax, wmin)

def gen_sample_dist_inf(dist, N_sample, mu=None, Sigma=None, w_min=None, w_max=None):
    if dist == "normal":
        w = normal(mu, Sigma, N=N_sample)
    elif dist == "uniform":
        w = uniform(w_min, w_max, N=N_sample)
    elif dist == "quadratic":
        w = quadratic(wmax=w_max, wmin=w_min, N=N_sample)
    else:
        raise ValueError("Unsupported distribution.")
    mean_ = np.mean(w, axis=1, keepdims=True)
    var_ = np.cov(w)
    return mean_, var_

def dr_kf_solve_measurement_update(Sigma_x_minus, C, Sigma_v_nom, theta, delta=1e-6):
    n = Sigma_x_minus.shape[0]
    m = Sigma_v_nom.shape[0]
    Sigma_v_var = cp.Variable((m, m), PSD=True)
    Z_var = cp.Variable((m, m), PSD=True)
    Sigma_x_var = cp.Variable((n, n), PSD=True)
    constraints = [
        cp.bmat([
            [Sigma_x_minus - Sigma_x_var, Sigma_x_minus @ C.T],
            [C @ Sigma_x_minus, C @ Sigma_x_minus @ C.T + Sigma_v_var]
        ]) >> 0,
        cp.bmat([
            [Sigma_v_nom, Z_var],
            [Z_var.T, Sigma_v_var]
        ]) >> 0,
        cp.trace(Sigma_v_var + Sigma_v_nom) - 2 * cp.trace(Z_var) <= theta**2,
        Sigma_v_var >> delta * np.eye(m)
    ]
    obj = cp.Maximize(cp.trace(Sigma_x_var))
    cp.Problem(obj, constraints).solve(solver=cp.MOSEK, verbose=False)
    return Sigma_v_var.value, Sigma_x_var.value

def is_detectable(A, C, tol=1e-6):
    n = A.shape[0]
    eigvals = np.linalg.eigvals(A)
    for lam in eigvals:
        if abs(lam) >= 1 - tol:
            M_detect = np.vstack([A - lam*np.eye(n), C])
            if np.linalg.matrix_rank(M_detect, tol=tol) < n:
                return False
    return True

def is_stabilizable(A, Sigma_w_nom, tol=1e-6):
    n = A.shape[0]
    try:
        B_nom = np.linalg.cholesky(Sigma_w_nom)
    except np.linalg.LinAlgError:
        return False
    eigvals = np.linalg.eigvals(A)
    for lam in eigvals:
        if abs(lam) >= 1 - tol:
            M_stab = np.hstack([A - lam*np.eye(n), B_nom])
            if np.linalg.matrix_rank(M_stab, tol=tol) < n:
                return False
    return True

def find_lyapunov_bound_v2(A, Sigma_w_nom):
    """
    Use bisection on a candidate scalar alpha to find M (PSD) such that:
         A M A^T + Sigma_w_nom << alpha M,
    with normalization trace(M)=1.
    Returns the minimal alpha found and the corresponding M.
    """
    n = A.shape[0]
    tol_bisect = 1e-3
    # initial bounds on alpha
    alpha_low = 0.0
    alpha_high = 10.0
    best_M = None
    best_alpha = None

    # First, enlarge alpha_high until feasibility is found.
    feasible = False
    while not feasible:
        M_var = cp.Variable((n, n), PSD=True)
        constraints = [
            cp.trace(M_var) == 1,
            alpha_high * M_var - A @ M_var @ A.T - Sigma_w_nom >> 1e-6 * np.eye(n)
        ]
        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            prob.solve(solver=cp.SCS, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            feasible = True
        else:
            alpha_high *= 2
            if alpha_high > 1e6:
                print("Bisection: Failed to find a feasible alpha; using fallback M = I.")
                return np.eye(n), None

    # Bisection loop to find the minimal feasible alpha.
    while alpha_high - alpha_low > tol_bisect:
        alpha_mid = (alpha_high + alpha_low) / 2.0
        M_var = cp.Variable((n, n), PSD=True)
        constraints = [
            cp.trace(M_var) == 1,
            alpha_mid * M_var - A @ M_var @ A.T - Sigma_w_nom >> 1e-6 * np.eye(n)
        ]
        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            prob.solve(solver=cp.SCS, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            # Feasible: try to lower alpha.
            alpha_high = alpha_mid
            best_M = M_var.value
            best_alpha = alpha_mid
        else:
            alpha_low = alpha_mid
    return best_M, best_alpha

def run_dr_kf_once(n=5, m=5, steps=2000, theta=0.5, tol=1e-4, N_samples=20, dist_type="normal"):
    # Regenerate system matrices until the nominal parameters satisfy detectability and stabilizability.
    while True:
        A = np.random.randn(n, n)
        # Generate process noise using nominal parameters.
        Wr = np.random.randn(n, n)
        Sigma_w_true = Wr @ Wr.T + 1e-4*np.eye(n)
        mu_w = np.zeros((n,1))
        _, Sigma_w_nom = gen_sample_dist_inf("normal", N_samples, mu=mu_w, Sigma=Sigma_w_true)
        Sigma_w_nom += 1e-4*np.eye(n)
        
        C = np.random.randn(m, n)
        
        # Generate measurement noise nominal covariance.
        Vr = np.random.randn(m, m)
        Sigma_v_true = Vr @ Vr.T + 1e-4*np.eye(m)
        mu_v = np.zeros((m,1))
        _, Sigma_v_nom = gen_sample_dist_inf("normal", N_samples, mu=mu_v, Sigma=Sigma_v_true)
        Sigma_v_nom += 1e-4*np.eye(m)
        
        if is_detectable(A, C) and is_stabilizable(A, Sigma_w_nom):
            break
        else:
            print("System did not satisfy detectability and stabilizability; regenerating...")
    
    # Compute the Lyapunov-type upper bound M using the nominal process noise.
    M, opt_alpha = find_lyapunov_bound_v2(A, Sigma_w_nom)
    if opt_alpha is None:
        print("Warning: Could not compute a proper Lyapunov bound; fallback M = I will be used.")
        M = np.eye(n)
    elif opt_alpha > 1 + 1e-6:
        print("Warning: The computed optimal alpha exceeds 1 (alpha = {:.4e}).".format(opt_alpha))
    else:
        print("Lyapunov inequality verified (optimal alpha = {:.4e} <= 1).".format(opt_alpha))
    
    # For further checking, verify the inequality directly:
    lyap_diff = M - A @ M @ A.T - Sigma_w_nom
    if np.all(np.linalg.eigvals(lyap_diff) >= -1e-6):
        print("Direct check: Lyapunov inequality holds for M.")
    else:
        print("Direct check: Lyapunov inequality does not hold for M.")
    
    Sigma_x_minus = np.eye(n)
    posterior_list = []
    conv_norms = []
    
    for step in range(steps):
        Sigma_v_sol, Sigma_x_sol = dr_kf_solve_measurement_update(Sigma_x_minus, C, Sigma_v_nom, theta)
        posterior_list.append(Sigma_x_sol)
        if step > 0:
            diff = np.linalg.norm(posterior_list[-1] - posterior_list[-2], 'fro')
            conv_norms.append(diff)
            if diff < tol:
                print(f"Convergence achieved at step {step+1} with diff {diff:.4e} (< tol {tol}).")
                break
        Sigma_x_minus = A @ Sigma_x_sol @ A.T + Sigma_w_nom

    # Check the upper-bound property: verify that M - X is PSD.
    check_Sigma_x = np.all(np.linalg.eigvals(M - posterior_list[-1]) >= -1e-6)
    check_Sigma_x_minus = np.all(np.linalg.eigvals(M - Sigma_x_minus) >= -1e-6)
    
    if check_Sigma_x:
        print("Final posterior covariance is upperbounded by M.")
    else:
        print("Final posterior covariance exceeds M!")
    
    if check_Sigma_x_minus:
        print("Final prior covariance is upperbounded by M.")
    else:
        print("Final prior covariance exceeds M!")
    
    return {
        "A": A,
        "Sigma_w_nom": Sigma_w_nom,
        "C": C,
        "Sigma_v_nom": Sigma_v_nom,
        "posterior_list": posterior_list,
        "conv_norms": conv_norms,
        "M": M,
        "opt_alpha": opt_alpha,
        "check_Sigma_x": check_Sigma_x,
        "check_Sigma_x_minus": check_Sigma_x_minus
    }

if __name__=="__main__":
    tol = 1e-4
    theta_values = [500, 1000]
    n_experiments = 10
    theta_summary = []
    
    for theta in theta_values:
        print(f"\n--- Testing for ambiguity set size theta = {theta} ---")
        conv_norms_all = []
        traces_final = []
        convergence_flags = []
        
        for exp_num in range(n_experiments):
            res = run_dr_kf_once(theta=theta, tol=tol)
            final_norm = res["conv_norms"][-1] if res["conv_norms"] else np.nan
            final_trace = np.trace(res["posterior_list"][-1])
            converged = (not np.isnan(final_norm)) and (final_norm < tol)
            convergence_flags.append(converged)
            conv_norms_all.append(final_norm)
            traces_final.append(final_trace)
            print(f"Experiment {exp_num+1}:")
            print(f"  Theta: {theta}")
            print(f"  Final conv norm: {final_norm:.4e}")
            print(f"  Final posterior trace: {final_trace:.4e}")
            print(f"  Convergence: {'YES' if converged else 'NO'}")
        
        avg_norm = np.mean([n for n in conv_norms_all if not np.isnan(n)])
        avg_trace = np.mean(traces_final)
        convergence_rate = np.mean(convergence_flags) * 100
        print(f"\nSummary for theta = {theta}:")
        print(f"  Average final convergence norm: {avg_norm:.4e}")
        print(f"  Average final posterior trace: {avg_trace:.4e}")
        print(f"  Convergence success rate: {convergence_rate:.1f}%")
        
        theta_summary.append({
            "theta": theta,
            "avg_norm": avg_norm,
            "avg_trace": avg_trace,
            "convergence_rate": convergence_rate
        })
    
    print("\n=== Final Summary Across All Theta Values ===")
    print("Theta\tConvergence Success Rate (%)\tAvg Final Norm\t\tAvg Posterior Trace")
    for summary in theta_summary:
        print(f"{summary['theta']:<8}\t{summary['convergence_rate']:<5.1f}\t\t\t{summary['avg_norm']:<18.4e}\t{summary['avg_trace']:<22.4e}")
