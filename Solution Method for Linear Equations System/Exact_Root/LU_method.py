import numpy as np

class LU:
    def __init__(self, A, b):
        self.A = np.array(A, dtype=float)  
        self.b = np.array(b, dtype=float)  
        self.n = len(b)
        self.L = np.zeros((self.n, self.n))
        self.U = np.zeros((self.n, self.n))
        
    def decompose(self):
        # Check if A is square and dimensions of A and b are compatible
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix A must be square")
            
        if len(self.b) != self.A.shape[0]:
            raise ValueError("Dimensions of A and b must be compatible")
            
        for i in range(self.n):
            # Set the diagonal of U to 1
            self.U[i][i] = 1.0
            
            # Calculate L matrix
            for j in range(i, self.n):
                self.L[j][i] = A[j][i]
                for k in range(i):
                    self.L[j][i] -= self.L[j][k] * self.U[k][i]
                    
            # Calculate U matrix
            for j in range(i, self.n):
                self.U[i][j] = A[i][j]
                for k in range(i):
                    self.U[i][j] -= self.L[i][k] * self.U[k][j]
                if abs(self.L[i][i]) < 1e-10:
                    raise ValueError("Cannot decompose: zero pivot encountered")
                self.U[i][j] /= self.L[i][i]
    
    def solve(self):
        # decompose if not already done
        if np.count_nonzero(self.L) == 0 or np.count_nonzero(self.U) == 0:
            self.decompose()
            
        y = np.zeros(self.n)
        x = np.zeros(self.n)
        
        # Forward substitution to solve Ly = b
        for i in range(self.n):
            y[i] = self.b[i]
            for j in range(i):
                y[i] -= self.L[i][j] * y[j]
            if abs(self.L[i][i]) < 1e-10:
                raise ValueError("Cannot solve system: zero on diagonal of L")
            y[i] /= self.L[i][i]
        
        # Back substitution to solve Ux = y
        for i in range(self.n-1, -1, -1):
            x[i] = y[i]
            for j in range(i+1, self.n):
                x[i] -= self.U[i][j] * x[j]
        return x
    
A = [[1, 1, -3, 2], [1, -2, 0, -1], [0, 1, 1, 3], [2, -3, 2, 0]]
b = [6, -6, 16, 6]
problem = LU(A, b)
solution = problem.solve()
print("Solution:", solution)
print("L matrix:\n", problem.L)
print("U matrix:\n", problem.U)
print("Verification A*x =", np.dot(problem.A, solution))
print("Verification L*U =\n", np.dot(problem.L, problem.U))
