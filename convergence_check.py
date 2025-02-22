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
            beta = 0.5*(a[j]+b[j])
            alpha = 12.0 / ((b[j]-a[j])**3)
            tmp = 3*x[i,j]/alpha - (beta - a[j])**3
            x[i,j] = beta + (tmp if tmp>=0 else -(-tmp)**(1./3.))**(1./3.)
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
        w = quadratic(w_max, w_min, N=N_sample)
    else:
        raise ValueError("Unsupported distribution.")
    mean_ = np.mean(w, axis=1, keepdims=True)
    var_  = np.cov(w)
    return mean_, var_

def dr_kf_solve_measurement_update(Sigma_x_minus, C, Sigma_v_nom, theta):
    n = Sigma_x_minus.shape[0]
    m = Sigma_v_nom.shape[0]
    Sigma_v_var = cp.Variable((m,m), PSD=True)
    Z_var       = cp.Variable((m,m), PSD=True)
    Sigma_x_var = cp.Variable((n,n), PSD=True)
    constraints = [
        cp.bmat([
            [Sigma_x_minus - Sigma_x_var, Sigma_x_minus@C.T],
            [C@Sigma_x_minus, C@Sigma_x_minus@C.T + Sigma_v_var]
        ]) >> 0,
        cp.bmat([
            [Sigma_v_nom, Z_var],
            [Z_var.T, Sigma_v_var]
        ]) >> 0,
        cp.trace(Sigma_v_var + Sigma_v_nom) - 2*cp.trace(Z_var) <= theta**2
    ]
    obj = cp.Maximize(cp.trace(Sigma_x_var))
    cp.Problem(obj, constraints).solve(solver=cp.MOSEK, verbose=False)
    return Sigma_v_var.value, Sigma_x_var.value

def make_stable_matrix(n, spectral_radius=0.95):
    A = np.random.randn(n, n)
    U, S, Vt = np.linalg.svd(A)
    A_stable = U @ np.diag((spectral_radius/np.max(S))*S) @ Vt
    return A_stable

def run_dr_kf_once(n=10, m=10, steps=50, theta=0.5, N_samples=20, dist_type="normal"):
    #A = make_stable_matrix(n, 0.9)
    A = np.random.randn(n, n)
    Wraw = np.random.randn(n,n)
    W = Wraw@Wraw.T + 1e-2*np.eye(n)
    C = np.random.randn(m,n)
    if dist_type=="normal":
        Vr = np.random.randn(m,m)
        true_Sigma_v = Vr@Vr.T + 1e-2*np.eye(m)
        mu_v = np.zeros((m,1))
        _, Sigma_v_nom = gen_sample_dist_inf("normal", N_samples, mu=mu_v, Sigma=true_Sigma_v)
    Sigma_v_nom += 1e-4*np.eye(m)
    O = control.obsv(A, C)
    det_ok = (np.linalg.matrix_rank(O) == n)
    B = np.linalg.cholesky(W)
    CC = control.ctrb(A, B)
    stab_ok = (np.linalg.matrix_rank(CC) == n)
    Sigma_x_minus = 0.01*np.eye(n)
    posterior_list = []
    for _ in range(steps):
        Sigma_v_sol, Sigma_x_sol = dr_kf_solve_measurement_update(Sigma_x_minus, C, Sigma_v_nom, theta)
        posterior_list.append(Sigma_x_sol)
        Sigma_x_minus = A@Sigma_x_sol@A.T + W
    monot_violations = []
    conv_norms = []
    for t in range(1, steps):
        diff = posterior_list[t] - posterior_list[t-1]
        e_diff = np.linalg.eigvalsh(0.5*(diff+diff.T))
        if np.min(e_diff) < -1e-9:
            monot_violations.append(t)
        conv_norms.append(np.linalg.norm(diff, 'fro'))
    return {
        "A": A, "W": W, "C": C, "Sigma_v_nom": Sigma_v_nom,
        "det_ok": det_ok, "stab_ok": stab_ok,
        "posterior_list": posterior_list,
        "monot_violations": monot_violations,
        "conv_norms": conv_norms
    }

if __name__=="__main__":
    for i in range(10):
        res = run_dr_kf_once()
        print(f"Experiment {i+1}: detect={res['det_ok']}, stab={res['stab_ok']}")
        print("Monotonicity violations at steps:", res["monot_violations"])
        print("Final conv norm =", res["conv_norms"][-1])
        print("Final posterior trace =", np.trace(res["posterior_list"][-1]))
