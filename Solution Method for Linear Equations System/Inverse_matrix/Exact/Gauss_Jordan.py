import numpy as np
from sympy import symbols, simplify
import pandas as pd

class Gauss_Jordan():
    def __init__(self, A):
        self.A = A.astype(float)
        self.n = self.A.shape[0]
        self.B = np.identity(self.n)
        self.AB = np.hstack((self.A, self.B))

    def solve(self):
        if np.linalg.det(self.A) == 0:
            raise ValueError("Matrix is degradation, no inverse")
        detA = np.linalg.det(self.A)
        print(f"detA = {detA}")
        for i in range(self.n):
            pivot = self.AB[i, i]
            index = i
            for j in range(i+1, self.n):
                if abs(self.AB[j][i]) > abs(pivot):
                    index = j
                    pivot = self.AB[i, i] 
                self.AB[index], self.AB[i] = self.AB[i].copy(), self.AB[index].copy() 
            if pivot == 0 :
                raise ValueError("Matrix is degradation, no inverse")
            self.AB[i] = self.AB[i] / pivot
            
            for j in range(self.n):
                if j != i:
                    self.AB[j] -= self.AB[j, i] * self.AB[i]
            print(self.AB)

        A_inv = self.AB[:, self.n:]
        return A_inv
        
# Example usage
# A = np.array([[1, 1, 1], [2, 3, 5], [4, 0, 5]])
# b = np.array([3, 10, 9])
A = np.array([[2, 1], [3, 5]])
solution = Gauss_Jordan(A)
A_inv = solution.solve()
print(np.round(A_inv, 4))