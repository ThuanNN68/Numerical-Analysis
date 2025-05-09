import numpy as np

class Power:
    def __init__(self, A, tol):
        self.A = A
        self.tol = tol
        self.n = len(A[0])