import numpy as np

class Danielevsky:
    def __init__(self, A):
        self.A = A
        self.n = len(A[0])

    def solve(self):
        A_current = self.A.copy()
        for i in range(self.n - 1):
            M_k = np.eye(self.n)
            for k in range(i + 1, self.n):
                if A_current[k, k] == 0:
                    pass
                M_k[i, k] = -A_current[i, k] / A_current[k, k]
            print("Matrix M in step", i + 1 ":" M_k)

            Mk_inv = np.eye(self.n)
            for j in range(self.n):
                if j != i:
                    Mk_inv[j] = M_k[j]
                else:
                    Mk_inv[j] = -M_k[j] / A_current[i, i]
            print("Inverse of M in step", i + 1 ":" Mk_inv)

            A_current = (M_k @ A_current) @ Mk_inv
            A_current = np.round(A_current, 6)
            print("Matrix A in step", i + 1 ":" A_current)

        return A_current
    
    
