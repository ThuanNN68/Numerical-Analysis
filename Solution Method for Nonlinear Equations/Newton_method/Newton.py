import numpy as np
import sympy as sp
import sys 
import scipy.optimize as opt 

class Newton_method:
    
    def __init__(self, f_expr, a, b, eps, iter):
        x = sp.symbols("x")
        f_expr = sp.sympify(f_expr)
        self.a = a
        self.b = b
        self.eps = eps
        self.iter = iter
        self.f = sp.lambdify(x, f_expr, "numpy")
        self.grad_f1 = sp.lambdify(x, sp.diff(f_expr, x), "numpy")
        self.grad_f2 = sp.lambdify(x, sp.diff(f_expr, x, 2), "numpy")

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
        """"Check the input"""
        # if a < b
        if(self.a >= self.b): return 0

        # if f(a) * f(b) >= 0
        if(self.f(self.a) * self.f(self.b) >= 0): return 0
        
        for h in (self.grad_f1, self.grad_f2):
            min_value, max_value = self.find_min_max(h, self.a, self.b)
            if min_value * max_value < 0:
                return 0
            
        return 1        

    def get_initial_guess(self):
        
        m1 = self.find_min_max(self.grad_f1, self.a, self.b)[0]
        M2 = self.find_min_max(self.grad_f2, self.a, self.b)[1]
        
        sign_gradf1 = 1 if m1 > 0 else -1
        sign_gradf2 = 1 if M2 > 0 else -1

        M = M2 / (2 * m1)
        
        if sign_gradf1 * sign_gradf2 > 0:
            c = self.b
        else:
            c = self.a
        return M, c
    
    def newton_with_iter(self):
        
        if self.check_input() == 0:
            return 0
        
        M, c = self.get_initial_guess()
        print(f"Initialization: x = {c}")
        for i in range(self.iter):
            d = c - self.f(c) / self.grad_f1(c)
            err = M*((d-c)**2)
            print(f"Iter {i+1}: x = {d}, err = {err}")
            c = d        
            
        return (c, err)
    
    def newton_with_eps(self):
        
        if self.check_input() == 0:
            return 0
        
        M, c = self.get_initial_guess()
        err = 1 + self.eps
        iter = 1
        print(f"Initialization: x = {c}")
        while M * self.f (c) / self.grad_f1(c) > self.eps:
            d = c - self.f(c) / self.grad_f1(c)
            err = M*((d-c)**2)
            print(f"Iter {iter}: x = {d}, err = {err}")
            c = d
            iter += 1
        return c, err

