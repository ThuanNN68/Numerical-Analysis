import numpy as np

class Bordering:
    def __init__(self, A):
        self.A = A.astype(float)
        self.n = self.A.shape[0]
    
    def solve(self):
        # Initialize with the first element
        if abs(self.A[0, 0]) < 1e-10:
            raise ValueError("Matrix A is not invertible")
        A_inv = np.array([[1 / self.A[0, 0]]])
        for k in range(1, self.n):
            Ak_inv = A_inv  # Current inverse
            u = self.A[:k, k].reshape(-1, 1)  # Column vector
            v = self.A[k, :k].reshape(1, -1)  # Row vector
            alpha = self.A[k, k] 
            
            # Calculate components for the block matrix
            z = Ak_inv @ u
            w = v @ Ak_inv
            s = alpha - v @ z
            
            if abs(s) < 1e-10:
                raise ValueError(f"Matrix A is not invertible at step {k}")

            # Calculate blocks for the expanded inverse
            B0 = Ak_inv + (z @ w) / s
            B1 = -z / s
            B2 = -w / s
            B3 = np.array([[1 / s]])
            
            A_inv = np.block([[B0, B1],
                              [B2, B3]])
        return A_inv

# Test with the given matrix
A = np.array([
    [-2, 3, 4, 3],
    [9, -10, 5, 7],
    [6, 7, 5, 4],
    [10, 9, -2, -10]
], dtype=float)
M = A.T @ A
try:
    problem = Bordering(M)
    M_inv = problem.solve()
except ValueError as e:
    print("Error:", e)
A_inverse = (M_inv @ A.T)
print("Inverse matrix:\n", np.round(A_inverse, 6))
print("Check A x A^-1 ≈ I:\n", np.round(A @ A_inverse, 6))