import numpy as np

class Jacobi:
    def __init__(self, A, x0, iter):
        self.A = A.astype(float)
        self.x0 = x0.astype(float)
        self.n = len(A[0])
        self.iter = iter
        
    def row_diagonally_dominant(self):
        for i in range(self.n):
            diagonal = abs(self.A[i, i])
            sum_row = sum(abs(self.A[i, j]) for j in range(self.n) if j != i)
            if diagonal <= sum_row:
                return False
        return True
    
    def col_diagonally_dominant(self):
        for i in range(self.n):
            diagonal = abs(self.A[i, i])
            sum_col = sum(abs(self.A[j, i]) for j in range(self.n) if j != i)
            if diagonal <= sum_col:
                return False
        return True
    
    def solve(self):
        # Check if the matrix is diagonally dominant
        if not self.row_diagonally_dominant() and not self.col_diagonally_dominant():
            raise ValueError("Matrix A is not diagonally dominant")

        # Find diagonal matrix T
        diag_elements = np.diag(self.A)
        T = np.diag(diag_elements)
        T_inv = np.diag(1.0 / diag_elements)
        
        x0 = self.x0.copy()
        x1 = np.eye(self.n)
        B = np.eye(self.n) - T_inv @ self.A
        for i in range(self.iter):
            x1 = B @ x0 + T_inv
            x0 = x1.copy()
            print(f"Iteration {i+1}: {x1}")
        return x1

A = np.array([
    [18, 3, 4, 3],
    [3, -15, 5, 4],
    [6, 2, 15, 4],
    [1, 3, -2, 10]
], dtype=float)
x0 = np.eye(np.shape(A)[0])
problem = Jacobi(A, x0, 20)
try:
    A_inv = problem.solve()
    print("Inverse matrix:\n", np.round(A_inv, 6))
    print("Check A x A^-1 ≈ I:\n", np.round(A @ A_inv, 4))
except ValueError as e:
    print("Erroer:", e)
            
            
        