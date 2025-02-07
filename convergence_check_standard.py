import numpy as np
import control

np.random.seed(42)

def make_stable_matrix(n, spectral_radius=0.95):
    A = np.random.randn(n, n)
    U, S, Vt = np.linalg.svd(A)
    A_stable = U @ np.diag((spectral_radius / np.max(S)) * S) @ Vt
    return A_stable

def run_standard_kf_once(n=10, m=10, steps=50):
    # Generate stable A
    A = make_stable_matrix(n, 0.9)
    # Process noise covariance W
    Wraw = np.random.randn(n,n)
    W = Wraw @ Wraw.T + 1e-2*np.eye(n)
    # Measurement matrix C
    C = np.random.randn(m,n)
    # Measurement noise covariance R
    Rraw = np.random.randn(m,m)
    R = Rraw @ Rraw.T + 1e-2*np.eye(m)
    
    # Check detectability
    O = control.obsv(A, C)
    det_ok = (np.linalg.matrix_rank(O) == n)
    # Check stabilizability
    B = np.linalg.cholesky(W)
    CC = control.ctrb(A, B)
    stab_ok = (np.linalg.matrix_rank(CC) == n)
    
    # Initialize prior covariance
    Sigma_x_minus = 10*np.eye(n)
    posteriors = []
    
    for _ in range(steps):
        # Standard Kalman filter measurement update:
        # Sigma_x_post = [ (Sigma_x_minus^-1) + C^T R^-1 C ]^-1
        # (Equivalent to Sigma_x_minus - K(...), but the inv form is neat.)
        inv_prior = np.linalg.inv(Sigma_x_minus)
        inv_R = np.linalg.inv(R)
        inv_post = inv_prior + C.T @ inv_R @ C
        Sigma_x_post = np.linalg.inv(inv_post)
        
        posteriors.append(Sigma_x_post)
        
        # Next prior
        Sigma_x_minus = A @ Sigma_x_post @ A.T + W
    
    # Check monotonicity among consecutive posteriors
    monot_violations = []
    conv_norms = []
    for t in range(1, steps):
        diff = posteriors[t] - posteriors[t-1]
        e_vals = np.linalg.eigvalsh(0.5*(diff + diff.T))
        if np.min(e_vals) < -1e-9:
            monot_violations.append(t)
        conv_norms.append(np.linalg.norm(diff, 'fro'))
    
    return {
        "A": A,
        "W": W,
        "C": C,
        "R": R,
        "det_ok": det_ok,
        "stab_ok": stab_ok,
        "posteriors": posteriors,
        "monot_violations": monot_violations,
        "conv_norms": conv_norms
    }

if __name__=="__main__":
    for i in range(3):
        res = run_standard_kf_once()
        print(f"\nExperiment {i+1}: detect={res['det_ok']}, stab={res['stab_ok']}")
        print("Monotonicity violations at steps:", res["monot_violations"])
        print("Final convergence norm =", res["conv_norms"][-1])
        print("Final posterior trace =", np.trace(res["posteriors"][-1]))
