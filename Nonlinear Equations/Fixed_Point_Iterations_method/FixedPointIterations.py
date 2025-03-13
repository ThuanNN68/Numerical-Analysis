import numpy as np
import sympy as sp
import scipy.optimize as opt

class FixedPointIter:
    
    def __init__(self, f, a, b, x_0, eps, iter):
        x = sp.symbols("x")
        self.f = sp.lambdify(x, f, "numpy")
        self.grad_f = sp.lambdify(x, sp.diff(f, x), "numpy")
        self.a = a
        self.b = b
        self.x_0 = x_0
        self.eps = eps
        self.iter = iter
        
    def find_min_max(self, f, a, b):
        try:
            
            min_result = opt.differential_evolution(f, bounds=[(a, b)])
            min_value = min_result.fun

            max_result = opt.differential_evolution(lambda x: -f(x), bounds=[(a, b)])
            max_value = -max_result.fun

            return min_value, max_value

        except Exception as e:
            print(f"Error in find_min_max: {e}")
            return None, None
        
    # Find q > 0 such that |f'(x)| <= q for all x in [a, b]
    def find_max_abs_gradf(self):
        g = lambda x: abs(self.grad_f(x))
        _, max_grad_f = self.find_min_max(g, self.a, self.b)
        return max_grad_f
    
    def check_input(self):
        """Check the input"""
        # Ensure a < b
        if self.a >= self.b:
            return 0
        
        # Ensure f(X) has value on interval [a, b]
        min_f, max_f = self.find_min_max(self.f, self.a, self.b)
        if (min_f <= self.a) or (max_f >= self.b): return 0

        # Ensure |f'(x)| < 1 for all x in [a, b]
        q = self.find_max_abs_gradf()
        if q is None or q >= 1:
            return 0
        
        return 1
    
    # Fixed point iteration with error tolerance eps
    def fixedpointiter_with_eps(self):
        
        if self.check_input() == 0:
            return 0
        
        # initialize parameters:
        iter_count = 1
        x_0 = self.x_0
        q = self.find_max_abs_gradf()
        err = self.eps + 1
        
        print(f"Initialization: x = {x_0}")
        # Evaluation 
        while err > self.eps:
            
            x_1 = self.f(x_0)
            err = q / (1 - q) * abs(x_1 - x_0)
            print(f"Iter {iter_count}: x = {x_1}, err = {err}")
            
            x_0 = x_1
            iter_count += 1
        
        return x_0, err 
    
    # Fixed point iteration with a fixed number of iterations
    def fixedpointiter_with_iter(self):
        
        if self.check_input() == 0:
            return 0
        
        # initialize parameters:
        x_0 = self.a
        q = self.find_max_abs_gradf()
        
        print(f"Initialization: x = {x_0}")
        
        # Evaluation 
        for i in range(self.iter):
            x_1 = self.f(x_0)    
            err = q / (1 - q) * abs(x_1 - x_0)
            print(f"Iter {i+1}: x = {x_1}, err = {err}")
            
            x_0 = x_1
        
        return x_0, err

        
        
        