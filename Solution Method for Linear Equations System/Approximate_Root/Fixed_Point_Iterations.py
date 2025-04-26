import numpy as np 

class fixed_point_iterations:
    def __init__(self, A, b, x0, eps):
        self.A = A
        self.b = b
        self.x0 = x0
        self.eps = eps
    
    def norm_matrix(self, A, p):
        return np.linalg.norm(A, p)
    def norm_vector(self, x, p):
        return np.linalg.norm(x, p)
    
    def check_input(self):
        if not isinstance(self.A, np.ndarray) or not isinstance(self.b, np.ndarray):
            raise ValueError("A and b must be numpy arrays")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix A must be square")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("Matrix A and vector b dimensions do not match")
        if len(self.x0) != self.A.shape[0]:
            raise ValueError("Initial guess x0 must have the same dimension as A")
    
    def solve(self):
        p = 1
        # Check ||A|| < 1
        if self.norm_matrix(self.A, p) >= 1:
            raise ValueError("The spectral radius of A must be less than 1 for convergence")
        norma = self.norm_matrix(self.A, p)
        print(f"Norm of A: {norma}")
        
        print(f"Initialization: x = {self.x0}")
        err = np.inf
        iter = 1
        x0 = self.x0.copy()
        
        while iter < 1000:
            x1 = np.dot(self.A, x0) + self.b
            err = abs(norma / (1 - norma))* self.norm_vector(x1 - x0, p)
            print(f"Iter {iter}: x = {x1}, err = {err}")
            x0 = x1.copy() 
            if err < self.eps:
                print(f"Converged in {iter} iterations")
                return x0, err
            iter += 1 
        return self.x0, err

A = np.array([[0, -0.4, 0.3],
     [-0.3, -0.1, 0.3],
     [-0.2, 0.1, -0.3]])
b = np.array([0.5, 0.4, 0.7])
x0 = np.zeros(3)
eps = 1e-6

solution = fixed_point_iterations(A, b, x0, eps)
solution.check_input()
result = solution.solve()
print(f"Final result: x = {result[0]}, err = {result[1]}")
