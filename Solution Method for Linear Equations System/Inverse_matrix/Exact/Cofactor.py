import numpy as np

class CofactorMatrix:
    def __init__(self, A):
        self.A = A
        self.n = A.shape[0]
        self.cofactor_matrix = np.zeros((self.n, self.n))
        
    def calculate_cofactor(self):
        for i in range(self.n):
            for j in range(self.n):
                minor = np.delete(np.delete(self.A, i, axis=0), j, axis=1)
                self.cofactor_matrix[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
        return self.cofactor_matrix
    
    def solve(self):
        cofactor_matrix = self.calculate_cofactor()
        adjugate_matrix = cofactor_matrix.T
        determinant = np.linalg.det(self.A)
        
        if determinant == 0:
            raise ValueError("The matrix is singular and cannot be inverted.")
        
        inverse_matrix = (1 / determinant) * adjugate_matrix
        return inverse_matrix
    
A = np.array([
    [-2, 3, 4, 3],
    [9, -10, 5, 7],
    [6, 7, 5, 4],
    [10, 9, -2, -10]
], dtype=float)

problem = CofactorMatrix(A)
try:
    A_inv = problem.solve()
    print("Inverse matrix:\n", np.round(A_inv, 6))
    print("Check A x A^-1 ≈ I:\n", np.round(A @ A_inv, 6))
except ValueError as e:
    print("Erroer:", e)