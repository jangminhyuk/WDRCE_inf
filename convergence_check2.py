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
        Sigma_v_var >> delta * np.eye(m)  # Enforce strict positive definiteness
    ]
    obj = cp.Maximize(cp.trace(Sigma_x_var))
    cp.Problem(obj, constraints).solve(solver=cp.MOSEK, verbose=False)
    return Sigma_v_var.value, Sigma_x_var.value

def make_stable_matrix(n, spectral_radius=0.95):
    A = np.random.randn(n, n)
    U, S, Vt = np.linalg.svd(A)
    A_stable = U @ np.diag((spectral_radius / np.max(S)) * S) @ Vt
    return A_stable

def run_dr_kf_once(n=10, m=10, steps=100, theta=0.5, N_samples=20, dist_type="normal"):
    # Regenerate system matrices until detectability and controllability are met.
    while True:
        A = np.random.randn(n, n)
        # Generate process noise nominal covariance using the nominal distribution
        Wr = np.random.randn(n, n)
        Sigma_w_true = Wr @ Wr.T + 1e-4 * np.eye(n)
        mu_w = np.zeros((n, 1))
        _, Sigma_w_nom = gen_sample_dist_inf("normal", N_samples, mu=mu_w, Sigma=Sigma_w_true)
        Sigma_w_nom += 1e-4 * np.eye(n)
        
        C = np.random.randn(m, n)
        
        # Generate measurement noise nominal covariance using the nominal distribution
        Vr = np.random.randn(m, m)
        Sigma_v_true = Vr @ Vr.T + 1e-4 * np.eye(m)
        mu_v = np.zeros((m, 1))
        _, Sigma_v_nom = gen_sample_dist_inf("normal", N_samples, mu=mu_v, Sigma=Sigma_v_true)
        Sigma_v_nom += 1e-4 * np.eye(m)
        
        # Check detectability and controllability assumptions.
        O = control.obsv(A, C)
        det_ok = (np.linalg.matrix_rank(O) == n)
        try:
            B = np.linalg.cholesky(Sigma_w_true)
        except np.linalg.LinAlgError:
            det_ok = False
        CC = control.ctrb(A, B)
        stab_ok = (np.linalg.matrix_rank(CC) == n)
        if det_ok and stab_ok:
            break  # Conditions met; proceed with this system.

    Sigma_x_minus = np.eye(n)
    posterior_list = []
    conv_norms = []

    for _ in range(steps):
        Sigma_v_sol, Sigma_x_sol = dr_kf_solve_measurement_update(Sigma_x_minus, C, Sigma_v_nom, theta)
        posterior_list.append(Sigma_x_sol)
        # Propagate state covariance using the nominal process noise covariance
        Sigma_x_minus = A @ Sigma_x_sol @ A.T + Sigma_w_nom

    # Compute differences between consecutive posteriors to check convergence.
    for t in range(1, steps):
        diff = posterior_list[t] - posterior_list[t-1]
        conv_norms.append(np.linalg.norm(diff, 'fro'))
    
    return {
        "A": A,
        "Sigma_w_nom": Sigma_w_nom,
        "C": C,
        "Sigma_v_nom": Sigma_v_nom,
        "det_ok": det_ok,
        "stab_ok": stab_ok,
        "posterior_list": posterior_list,
        "conv_norms": conv_norms
    }

if __name__=="__main__":
    # Convergence tolerance: final convergence norm below this value implies convergence.
    tol = 1e-4

    # Define a list of theta values to test.
    theta_values = [0.1, 1, 5, 10, 50, 100, 500, 1000, 10000]
    n_experiments = 20  # Number of experiments for each theta.
    
    # This list will store summary results for each theta.
    theta_summary = []
    
    for theta in theta_values:
        print(f"\n--- Testing for ambiguity set size theta = {theta} ---")
        conv_norms_all = []
        traces_final = []
        convergence_flags = []
        
        for exp_num in range(n_experiments):
            res = run_dr_kf_once(theta=theta)
            final_norm = res["conv_norms"][-1]
            final_trace = np.trace(res["posterior_list"][-1])
            converged = final_norm < tol
            convergence_flags.append(converged)
            
            conv_norms_all.append(final_norm)
            traces_final.append(final_trace)
            
            print(f"Experiment {exp_num+1}:")
            print(f"  Theta: {theta}")
            print(f"  Final conv norm: {final_norm:.4e}")
            print(f"  Final posterior trace: {final_trace:.4e}")
            print(f"  Convergence: {'YES' if converged else 'NO'}")
        
        # Summary for the current theta value.
        avg_norm = np.mean(conv_norms_all)
        avg_trace = np.mean(traces_final)
        convergence_rate = np.mean(convergence_flags) * 100
        print(f"\nSummary for theta = {theta}:")
        print(f"  Average final convergence norm: {avg_norm:.4e}")
        print(f"  Average final posterior trace: {avg_trace:.4e}")
        print(f"  Convergence success rate: {convergence_rate:.1f}% of experiments converged")
        
        theta_summary.append({
            "theta": theta,
            "avg_norm": avg_norm,
            "avg_trace": avg_trace,
            "convergence_rate": convergence_rate
        })
    
    # Final summary across all theta values.
    print("\n=== Final Summary Across All Theta Values ===")
    print("Theta\tConvergence Success Rate (%)\tAvg Final Norm\t\tAvg Posterior Trace")
    for summary in theta_summary:
        print(f"{summary['theta']:<8}\t{summary['convergence_rate']:<5.1f}\t\t\t{summary['avg_norm']:<18.4e}\t{summary['avg_trace']:<22.4e}")
