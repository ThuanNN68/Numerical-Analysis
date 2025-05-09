import numpy as np

class SVDs:
    def __init__(self, A, tol):
        self.A = A
        self.n = len(A[0])
        self.tol = tol
        