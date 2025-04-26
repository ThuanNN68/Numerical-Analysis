import numpy as np

class Jacobi:
    def __init__(self, A, b, x0, eps):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.x0 = x0
        self.eps = eps
        self.n = len(A[0])        
        
    def norm_matrix(self, A, p):
        return np.linalg.norm(A, p)
    
    def norm_vector(self, x, p):
        return np.linalg.norm(x, p)
        
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
        
        # Check input
        if not isinstance(self.A, np.ndarray) or not isinstance(self.b, np.ndarray):
            raise ValueError("A and b must be numpy arrays")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix A must be square")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("Matrix A and vector b dimensions do not match")
        if len(self.x0) != self.A.shape[0]:
            raise ValueError("Initial guess x0 must have the same dimension as A")
        
        # Find dianogal matrix T
        diag_elements = np.diag(self.A)
        T = np.diag(diag_elements)
        T_inv = np.diag(1.0 / diag_elements)
        
        if self.row_diagonally_dominant():
            print ("Matrix A is row diagonally dominant")
            p = np.inf
            lambda0 = 1
            q = 0
            for i in range(self.n):
                sum_row = sum(abs(self.A[i, j]) for j in range(self.n) if j != i)
                if q < sum_row / self.A[i][i]:
                    q = sum_row / self.A[i][i]
            
        elif self.col_diagonally_dominant():
            print ("Matrix A is column diagonally dominant")
            p = 1
            lambda0 = np.max(diag_elements) / np.min(diag_elements)
            q = 0 
            for j in range(self.n):
                sum_row = sum(abs(self.A[i, j]) for i in range(self.n) if i != j)
                if q < sum_row / self.A[j][j]:
                    q = sum_row / self.A[j][j]

        iter = 1
        x0 = self.x0.copy()
        x1 = np.zeros(self.n)
        while iter < 1000:
            for i in range(self.n):
                sum_ = 0
                for j in range(self.n):
                    if i != j:
                        sum_ += self.A[i][j] * x0[j]
                x1[i] = (self.b[i] - sum_) / self.A[i][i]
            
            err = self.norm_vector(x1 - x0, p) * (lambda0 * q) / (1 - q)
            print(f"Iter {iter}: x: {x1}, err: {err}")
            
            if err < self.eps:
                return x1, err
            x0 = x1.copy()
            iter += 1
        return None, None

# Example usage:
A = np.array([[10, 2, -3], [1, 15, 5], [3, -4, 20]], dtype=float)
b = np.array([1, 2, 3], dtype=float)
x0 = np.array([0.5, -0.2, 0.3])

problem = Jacobi(A, b, x0, 1e-6)

x, err = problem.solve()
print("Solution:", x)
print("Error:", err)
print("Verification A*x:", np.dot(A, x))
print("Original b:", b)

                
        
        