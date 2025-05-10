import numpy as np

class Choleski():
    def __init__(self, A):
        self.A = A
        self.n = len(A)
        self.L = np.zeros((self.n, self.n))
        self.M = A.T @ A

    # Choleski decomposition
    def decompose(self, A):
        # M is symmetric and positive definite
        
        # if not np.all(np.linalg.eigvals(A) > 0):
            # raise ValueError("Matrix A must be positive definite")
        # for i in range(self.n):
            # if np.linalg.det(A[:i,:i]) == 0:
                # raise ValueError("Matrix A is not positive definite")
            
        # A = L^T * L 
        # L is lower triangular matrix
        # L^T = L is upper triangular matrix
        L = np.zeros((self.n, self.n))
        for i in range(self.n):
            diagval = A[i][i]
            for j in range(i):
                diagval -= L[i][j] ** 2
            L[i][i] = np.sqrt(diagval)
            
            for j in range(i+1, self.n):
                sumval = A[i][j]
                for k in range(i):
                    sumval -= L[i][k] * L[j][k]
                L[j][i] = sumval / L[i][i]
        return L
                
    def solve_m_matrix (self):
        self.L = self.decompose(self.M)

        L_inv = np.zeros((self.n, self.n))
        I = np.identity(self.n)

        for i in range(self.n):
            # Forward substitution: L x = e_i
            x = np.zeros(self.n)
            for j in range(self.n):
                sum_val = I[j][i]
                for k in range(j):
                    sum_val -= self.L[j][k] * x[k]
                x[j] = sum_val / self.L[j][j]
            
            L_inv[:, i] = x
        M_inv = L_inv.T @ L_inv
        return M_inv
    
    def solve_a_matrix(self):
        M_inv = self.solve_m_matrix()
        A_inv = M_inv @ self.A.T 
        return A_inv
    
A = np.array([
    [-2, 3, 4, 3],
    [9, -10, 5, 7],
    [6, 7, 5, 4],
    [10, 9, -2, -10]
], dtype=float)
problem = Choleski(A)
M_inv = problem.solve_m_matrix()
A_inv = problem.solve_a_matrix()

print("Verification L*L^T\n", np.dot(problem.L, problem.L.T))
print("M matrix: \n", np.round(problem.M, 6))
print("Inverse M matrix:\n", np.round(M_inv, 6))
print("Check M x M^-1 ≈ I:\n", np.round(problem.M @ M_inv, 6))
print("Inverse A matrix:\n", np.round(A_inv, 6))
print("Check A x A^-1 ≈ I:\n", np.round(A @ A_inv, 6))

    
    
        
