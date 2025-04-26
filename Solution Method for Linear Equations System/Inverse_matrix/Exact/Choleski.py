import numpy as np

class Choleski():
    def __init__(self, A):
        self.A = A
        self.n = len(A)
        self.L = np.zeros((self.n, self.n))

    # Choleski decomposition
    def decompose(self):
        # Check if A is symmetric and positive definite
        if not np.allclose(self.A, self.A.T):
            raise ValueError("Matrix A must be symmetric")
        
        # if not np.all(np.linalg.eigvals(self.A) > 0):
            # raise ValueError("Matrix A must be positive definite")
        for i in range(self.n):
            if np.linalg.det(self.A[:i,:i]) == 0:
                # raise ValueError("Matrix A is not positive definite")
                pass
        # A = L * L^T 
        # L is lower triangular matrix
        # L^T is upper triangular matrix
        for i in range(self.n):
            diagval = self.A[i][i]
            for j in range(i):
                diagval -= self.L[i][j] ** 2
            if diagval <= 0:
                raise ValueError("Matrix A is not positive definite")
            self.L[i][i] = np.sqrt(diagval)
            
            for j in range(i+1, self.n):
                sumval = self.A[i][j]
                for k in range(i):
                    sumval -= self.L[i][k] * self.L[j][k]
                self.L[j][i] = sumval / self.L[i][i]
                
    def solve(self):
        # decompose if not already done
        if np.count_nonzero(self.L) == 0:
            self.decompose()

        inv_A = np.zeros((self.n, self.n))
        E = np.identity(self.n)
        
        for i in range(self.n):
            y = np.zeros(self.n)
            # Forward substitution to solve Ly = E
            for j in range(self.n):
                y[j] = E[j][i]
                for k in range(j):
                    y[j] -= self.L[j][k] * y[k]
                y[j] /= self.L[j][j]
            x = np.zeros(self.n)
            # Back substitution to solve L^T x = y
            for j in reversed(range(self.n)):
                x[j] = y[j]
                for k in range(j+1, self.n):
                    x[j] -= self.L[k][j] * x[k]  # L^T[j][k] = L[k][j]
                x[j] /= self.L[j][j]

            inv_A[:, i] = x
        return inv_A
    
A = np.array([
    [-2, 3, 4, 3],
    [9, -10, 5, 7],
    [6, 7, 5, 4],
    [10, 9, -2, -10]
], dtype=float)
M = A * A.T
problem = Choleski(M)
solution = problem.solve()

print("Solution:", solution)
print("L matrix:\n", problem.L)
print("Verification A*x =", np.round(np.dot(problem.A, solution)))
print("Verification L*L^T =\n", np.dot(problem.L, problem.L.T))
print("Original A =\n", A)
print("Verification A*A_inv =\n", np.round(np.dot(A, solution), 4))

A_inv = (solution @ A.T)
print(A_inv)
    
    
        
