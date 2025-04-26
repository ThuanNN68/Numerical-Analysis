import numpy as np

class Newton:
    def __init__(self, A, x0, p, iter, eps):
        self.A = A.astype(float)
        self.x0 = x0.astype(float)
        self.n = len(A[0])
        self.p = p
        self.iter = iter
        self.eps = eps
    
    def norm_matrix(self, A):
        return np.linalg.norm(A, self.p)   
    
    def solve(self):
        # Check the convergence of the method
        R0 = np.eye(self.n) - self.A @ self.x0
        normR0 = self.norm_matrix(R0)
        if normR0 >= 1:
            raise ValueError("Matrix R0 is not convergent")
        normx0 = self.norm_matrix(self.x0)
        xk = self.x0.copy()
        err = np.inf
        for k in range(self.iter):
            # Newton iteration: X_{k+1} = X_k + X_k @ (I - A @ X_k)
            # Or equivalently: X_{k+1} = X_k @ (2*I - A @ X_k)
            # We already have Rk = I - A @ Xk
            xk_new = xk @ (2 * np.eye(self.n) - self.A @ xk)

            # Calculate the new residual for the next iteration and for checking
            Rk = np.eye(self.n) - self.A @ xk_new
            normRk_new = self.norm_matrix(Rk)

            # Calculate change in estimate
            norm_diff = self.norm_matrix(xk_new - xk)
            xk = xk_new
            R0 = Rk # Update residual for next iteration
            err = normx0 * normR0 ** (2 ** (k + 1)) / (1 - normR0)    
            print(f"Iteration {k+1}: {np.round(xk, 6)}")
            print(f"Residual norm: {np.round(normRk_new)}, Change in estimate: {np.round(norm_diff)}")
            print(f"Error: {np.round(err, 6)}")
            if err < self.eps:
                break
        return xk
    
A = np.array([
    [4, 1, 1],
    [1, 5, 2],
    [1, 2, 6]
], dtype=float)
n = np.shape(A)[0]

row_diagonally_dominant = True
for i in range(n):
    diagonal = abs(A[i, i])
    sum_row = sum(abs(A[i, j]) for j in range(n) if j != i)
    if diagonal <= sum_row:
        row_diagonally_dominant = False
        break
    
if not row_diagonally_dominant:
    norm_A_1 = np.linalg.norm(A, 1)
    norm_A_inf = np.linalg.norm(A, np.inf)
    x0 = A.T / (norm_A_1 * norm_A_inf)
    print("Using Scaled Transpose as Initial Guess (x0):")
    print(np.round(x0, 5))
else:
    print("Matrix A is row diagonally dominant")
    norm_A_inf = np.linalg.norm(A, np.inf)
    x0 = np.eye(n) / norm_A_inf
    print("Using Identity Matrix as Initial Guess (x0):")
    print(np.round(x0, 5))
    
iter = 10
p = 1
eps = 1e-6
problem = Newton(A, x0, p, iter, eps)

try:
    A_inv = problem.solve()
    print("Inverse matrix:\n", np.round(A_inv, 4))
    print("Check A x A^-1 ≈ I:\n", np.round(A @ A_inv, 4))
except ValueError as e:
    print("Error:", e)