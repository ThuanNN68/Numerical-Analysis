import numpy as np 
import sympy as sp

class Bisection_method:
    
    def __init__(self, f, a, b, eps, iter):
        x = sp.symbols("x")
        self.f = sp.lambdify(x, f, "numpy")
        self.a = a
        self.b = b
        self.eps = eps
        self.iter = iter
    
    def check_input(self):
        """"Check the input"""
        # if a < b
        if self.a > self.b: return 0

        # if f(a) * f(b) < 0
        if self.f(self.a) * self.f(self.b) >= 0: return 0
        
        return 1
    
    # Evaluate the error number 
    def evaluate_error(self, a_0, b_0):
        return np.abs(a_0 - b_0)
        
    # Bisection method with error tolerance eps
    def bisection_with_eps(self):
        
        if self.check_input() == 0:
            return 0
        
        # initialize parameters:
        iter_count = 0
        positive = 1 if self.f(self.a) > 0 else -1 
        
        # Evalution 
        while abs(self.a - self.b) >= self.eps:
            
            mid = (self.a + self.b) /2
            err = self.evaluate_error(self.a, self.b)
            
            if self.f(mid) == 0:
                return mid
            print(f"Iter {iter_count}: a = {self.a}, b = {self.b}, mid = {mid}, f(mid) = {'+' if self.f(mid) > 0 else '-'}, err = {err}")
            
            if (self.f(mid) * positive) > 0:
                self.a = mid
            else:
                self.b = mid
                
            iter_count += 1
          
        return mid, err

    # Bisection method with a fixed number of iterations
    def bisection_with_iter(self):
        
        if self.check_input() == 0:
            return 0
        
        # if a or b is a root
        if self.f(self.a) == 0: 
            return self.a
        if self.f(self.b) == 0: 
            return self.b
        
        # initialize parameters:
        positive = 1 if self.f(self.a) > 0 else -1 
        mid = (self.a + self.b) /2
    
        # Evalution 
        for i in range(self.iter + 1):
            mid = (self.a + self.b) /2
            err = self.evaluate_error(self.a, self.b)
            if self.f(mid) == 0:
                return mid
            
            print(f"Iter {i}: a = {self.a}, b = {self.b}, mid = {mid}, f(mid) = {'+' if self.f(mid) > 0 else '-'}, err = {err}")
            
            if(self.f(mid) * positive) > 0:
                self.a = mid
            else:
                self.b = mid

        return mid, err
