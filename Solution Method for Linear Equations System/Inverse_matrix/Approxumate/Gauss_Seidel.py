import numpy as np

class GaussSeidel:
    def __init__(self, A, x0, iter):
        self.A = A.astype(float)
        self.x0 = x0.astype(float)
        self.E = np.eye(A.shape[0])
        self.n = self.A.shape[0]
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
        if not self.row_diagonally_dominant() and not self.col_diagonally_dominant():
            raise ValueError("Matrix A is not diagonally dominant")
        T = np.diag(np.diag(self.A))
        T_inv = np.diag(1.0 / np.diag(self.A))
        B = np.eye(self.n) - T_inv @ self.A
        x0 = self.x0.copy()
        x1 = np.eye(self.n)
        
        L = np.zeros((self.n, self.n))
        U = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i > j:
                    L[i, j] = self.A[i, j]
                elif i <= j:
                    U[i, j] = self.A[i, j]
        print("L:\n", L)
        print("U:\n", U)
        for k in range(self.iter):
            for i in range(self.n):
                for j in range(self.n):
                    sum_L = 0
                    for p in range(i):
                        sum_L += self.A[i, p] * x1[p, j]
                    
                    sum_U = 0
                    for p in range(i + 1, self.n):
                        sum_U += self.A[i, p] * x0[p, j]
                    # Update x1
                    x1[i, j] = (self.E[i, j] - sum_L - sum_U) / self.A[i, i]
            x0 = x1.copy()
            print(f"Iteration {k+1}: {x1}")
        return x1
    
A = np.array([
    [17, 3, 4, 3],
    [3, -15, 5, 4],
    [6, 2, 15, 4],
    [1, 3, -2, 10]
], dtype=float)
x0 = np.eye(np.shape(A)[0])
iter = 10

problem = GaussSeidel(A, x0, iter)
try:
    A_inv = problem.solve()
    print("Inverse matrix:\n", np.round(A_inv, 4))
    print("Check A x A^-1 ≈ I:\n", np.round(A @ A_inv, 4))
except ValueError as e:
    print("Erroer:", e)
                    
            
        
        
    