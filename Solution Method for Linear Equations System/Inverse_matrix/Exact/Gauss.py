import numpy as np

class Gauss:
    def __init__(self, A):
        self.A = A
        self.n = A.shape[0]
        self.Ab = np.hstack((A, np.eye(self.n)))  # Augmented matrix [A|I]
        self.epsilon = 1e-10
        
    def pivot(self, i):
        if abs(self.Ab[i, i]) >= self.epsilon:
            return True
        for l in range(i + 1, self.n):
            if abs(self.Ab[l, i]) > self.epsilon:
                self.Ab[[i, l],:] = self.Ab[[l, i],:]
                return True
        return False
    
    def solve(self):
        row = 0
        pivot_columns = [] # list to store pivot columns
        pivot_rows = []    # list to store corresponding pivot rows
        
        # Forward elimination
        for col in range(self.n):
            if row >= self.n:
                break
            if not self.pivot(row):
                print(f"Matrix is singular: no non-zero pivot found in column {col}")
                return None
            pivot_columns.append(col)
            pivot_rows.append(row)
            pivot_element = self.Ab[row, col]
            print(f"Pivot element at row {row}, col {col}: {pivot_element}")
            
            # Eliminate entries below the pivot
            for k in range(row + 1, self.n):
                if abs(self.Ab[k, col]) > self.epsilon:
                    factor = self.Ab[k, col] / pivot_element
                    self.Ab[k, col:] -= factor * self.Ab[row, col:]
                    self.Ab[k, col] = 0.0  # Explicitly set to zero to avoid numerical errors
            
            print(np.array2string(self.Ab, formatter={'float_kind': lambda x: "%.6f" % x}))
            row += 1
        
        print("After forward elimination:")
        print(np.array2string(self.Ab, formatter={'float_kind': lambda x: "%.6f" % x}))
        
        # Check for inconsistent system
        rankA = len(pivot_columns)
        if rankA < self.n:
            print(f"Matrix is singular with rank {rankA}")
            return None
        
        # Back substitution
        for i in range(self.n - 1, -1, -1):
            # Normalize the pivot row
            pivot = self.Ab[i, pivot_columns[i]]
            self.Ab[i, :] /= pivot
            
            # Eliminate entries above the pivot
            for k in range(i - 1, -1, -1):
                factor = self.Ab[k, pivot_columns[i]]
                self.Ab[k, :] -= factor * self.Ab[i, :]
        
        # Extract the inverse matrix from the right side of the augmented matrix
        A_inv = self.Ab[:, self.n:]
        
        print("After back substitution:")
        print(np.array2string(A_inv, formatter={'float_kind': lambda x: "%.6f" % x}))
        return A_inv

# Test with the given matrix
if __name__ == "__main__":
    A = np.array([
        [-2, 3, 4, 3],
        [9, -10, 5, 7],
        [6, 7, 5, 4],
        [10, 9, -2, -10]
    ], dtype=float)
    
    print("Original matrix A:")
    print(A)
    
    problem = Gauss(A)
    A_inv = problem.solve()
    
    if A_inv is not None:
        print("Inverse matrix:\n", np.round(A_inv, 6))
        print("Check A x A^-1 ≈ I:\n", np.round(A @ A_inv, 6))
        
        # Verify with NumPy's built-in inverse function
        print("\nVerify with NumPy's inverse:")
        np_inv = np.linalg.inv(A)
        print(np.round(np_inv, 6))