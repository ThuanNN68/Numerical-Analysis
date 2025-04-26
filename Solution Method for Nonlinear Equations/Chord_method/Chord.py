import numpy as np
import sympy as sp
import scipy.optimize as opt 

class Chord_method:
    
    def __init__(self, f_expr, a, b, eps, iter):
        x = sp.symbols("x")
        self.f = sp.lambdify(x, f_expr, "numpy")
        self.grad_f1 = sp.lambdify(x, sp.diff(f_expr, x), "numpy")
        self.grad_f2 = sp.lambdify(x, sp.diff(f_expr, x, 2), "numpy")
        self.a = a
        self.b = b
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
    
    def check_input(self):
        """Check the input"""
        # if a < b
        if self.a > self.b: return 0

        # if f(a) * f(b) < 0
        if self.f(self.a) * self.f(self.b) >= 0: return 0
        
        for h in (self.grad_f1, self.grad_f2):
            min_value, max_value = self.find_min_max(h, self.a, self.b)
            if min_value * max_value < 0:
                return 0
            
        return 1   
    
    def get_initial_guess(self):
        
        m1, M1 = self.find_min_max(self.grad_f1, self.a, self.b)
        M = abs((M1-m1)/m1)
        m2 = self.find_min_max(self.grad_f2, self.a, self.b)[0]
        
        if (m1 * m2 > 0):
            d, x_0 = self.b, self.a
        else:
            d, x_0 = self.a, self.b
        return d, x_0, M
    
    # Chord method with fixed number of iterations
    def chord_with_iter(self):
        if self.check_input() == 0:
            return 0

        d, x_0, M = self.get_initial_guess()
        fd = self.f(d)

        print(f"Initialization: x = {x_0}")
        
        for i in range(self.iter):
            fx_0 = self.f(x_0)
            x_1 = x_0 - fx_0 * (d - x_0) / (fd - fx_0)
            
            # evaluate the error
            err = M * abs(x_1 - x_0)
            print(f"Iteration {i+1}: x = {x_1}, err = {err}")
            x_0 = x_1
            
        return x_0, err
    
    # Chord method with error tolerance eps
    def chord_with_eps(self):
        if self.check_input() == 0:
            return 0
        
        d, x_0, M = self.get_initial_guess()
        fd = self.f(d)
        err = self.eps + 1
        iter = 1
        print(f"Initialization: x = {x_0}")
        while err > self.eps:
            fx_0 = self.f(x_0)
            x_1 = x_0 - fx_0 * (d - x_0) / (fd - fx_0)
            
            # evaluate the error
            err = M * abs(x_1 - x_0)
            print(f"Iteration {iter}: x = {x_1}, err = {err}")
            #update parameters
            iter += 1
            x_0 = x_1
            
        return x_0, err

