import numpy as np

class Choleski:
    def __init__(self, A, b):
        self.A = np.array(A, dtype=float)  
        self.b = np.array(b, dtype=float)  
        self.n = len(b)
        self.L = np.zeros((self.n, self.n))
    
    def decompose(self):
        # Check if A is symmetric and positive definite
        if not np.allclose(self.A, self.A.T):
            raise ValueError("Matrix A must be symmetric")
        
        if not np.all(np.linalg.eigvals(self.A) > 0):
            raise ValueError("Matrix A must be positive definite")
        
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
            
        y = np.zeros(self.n)
        x = np.zeros(self.n)
        
        # Forward substitution to solve Ly = b
        for i in range(self.n):
            y[i] = self.b[i]
            for j in range(i):
                y[i] -= self.L[i][j] * y[j]
            y[i] /= self.L[i][i]
        
        # Back substitution to solve L^T x = y
        for i in range(self.n-1, -1, -1):
            x[i] = y[i]
            for j in range(i+1, self.n):
                x[i] -= self.L[j][i] * x[j] # L^T[j][i] = L[i][j]
            x[i] /= self.L[i][i]
        return x
    
A = [[4, 12, -16], 
     [12, 37, -43], 
     [-16, -43, 98]]
b = [8, 38, -30]

problem = Choleski(A, b)
solution = problem.solve()

print("Solution:", solution)
print("L matrix:\n", problem.L)
print("Verification A*x =", np.dot(problem.A, solution))
print("Verification L*L^T =\n", np.dot(problem.L, problem.L.T))
print("Original A =\n", problem.A)