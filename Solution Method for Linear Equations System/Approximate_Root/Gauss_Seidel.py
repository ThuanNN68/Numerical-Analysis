import numpy as np

class GaussSeidel:
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
        
        if self.row_diagonally_dominant():
            print ("Matrix A is row diagonally dominant")
            p = np.inf
            s= 0
            phi = -np.inf
            for i in range(self.n):
                sum_up = 0
                sum_low = 0
                for j in range(self.n):
                    if j < i:
                        sum_low += abs(self.A[i][j])
                    elif j > i:
                        sum_up += abs(self.A[i][j])
                if phi < sum_low / (self.A[i][i] - sum_up):
                    phi = sum_low / (self.A[i][i] - sum_up)
            print ("phi = ", phi)
        elif self.col_diagonally_dominant():
            print ("Matrix A is column diagonally dominant")
            p = 1
            s = 0
            phi = -np.inf
            for j in range(self.n):
                sum_up = 0
                sum_low = 0
                for i in range(self.n):
                    if i < j:
                        sum_up += abs(self.A[i][j])
                    elif i > j:
                        sum_low += abs(self.A[i][j])
                if s < sum_low / self.A[j][j]:
                    s = sum_low / self.A[j][j]
                if phi < sum_up / (self.A[j][j] - sum_low):
                    phi = sum_up / (self.A[j][j] - sum_low)
            print (f"phi = {phi}")
            print (f"s = {s} ")
        
        x0 = self.x0.copy()
        x1 = np.zeros(self.n)
        iter = 1
        k_err = phi / ((1 - phi) * (1 - s))
        while iter <= 1000:
            for i in range(self.n):
                sum1 = sum(self.A[i][j] * x1[j] for j in range(0, i))
                sum2 = sum(self.A[i][j] * x0[j] for j in range(i + 1, self.n))
                x1[i] = (self.b[i] - sum1 - sum2) / self.A[i][i]
            err = k_err * self.norm_vector(x1 - x0, np.inf)
            print(f"Iter {iter}: x1 = {x1}, err = {err}")
            x0 = x1.copy()
            if err < self.eps:
                print(f"Converged in {iter} iterations")
                return x0, err
            iter += 1
        print("Failed to converge")
        return None, None
        
# Example usage:
A = np.array([[10, 2, -3], [1, 15, 5], [3, -4, 20]], dtype=float)
b = np.array([1, 2, 3], dtype=float)
x0 = np.array([0.5, -0.2, 0.3])

problem = GaussSeidel(A, b, x0, 1e-6)

x, err = problem.solve()
print("Solution:", x)
print("Error:", err)
print("Verification A*x:", np.dot(A, x))
print("Original b:", b)
     
