import numpy as np
import sympy as sp

class FixedPointIter2Equations:
    def __init__(self, f1, f2, q, a, b, c, d, x1, x2, iter):
        # Define symbolic variables
        self.x1, self.x2 = sp.symbols('x1 x2')

        # Convert sympy expressions to numpy functions using lambdify
        self.f1 = sp.lambdify((self.x1, self.x2), f1, "numpy")
        self.f2 = sp.lambdify((self.x1, self.x2), f2, "numpy")

        # Store parameters
        self.q = q
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x1 = x1
        self.x2 = x2
        self.iter = iter
        
    def find_err(self, iterations):
        # Compute the error based on the given formula and round it
        return round(self.q**iterations * ((self.b - self.a) + (self.d - self.c)), 8)
    
    def solve_fixed_iters(self):
        print(f"Initialization: x_1 = {self.x1}, x_2 = {self.x2}")

        # Initial values (ensure they are floats)
        x1_k = float(self.x1)
        x2_k = float(self.x2)
        err = 0
        
        for i in range(self.iter):
            # Compute the next values using the lambdified functions
            x1_next = self.f1(x1_k, x2_k)
            x2_next = self.f2(x1_k, x2_k)
            
            # Ensure the results are floats and calculate error
            x1_next = float(x1_next)
            x2_next = float(x2_next)    
            
            # Compute the error for this iteration
            err = self.find_err(i+1)
            print(f"Iter {i+1}: x_1 = {round(x1_next, 8)}, x_2 = {round(x2_next, 8)}, err = {round(err, 8)}")
            
            # Update the values for the next iteration
            x1_k = x1_next
            x2_k = x2_next
            
        return x1_k, x2_k, err
    
# Example for test
x1, x2 = sp.symbols("x1 x2")
f1 = 1/sp.sqrt(2) * sp.sqrt(x1*x2 + 5*x1 - 1)
f2 = sp.sqrt(x1 + 3*sp.log(x1)/sp.log(10))
q = 0.857
a, b, c, d = 3.3, 3.5, 2.2, 2.3
x1_0, x2_0 = 3.3, 2.2
iter = 10
solution = FixedPointIter2Equations(f1, f2, q, a, b, c, d, x1_0, x2_0, iter)
x1_sol, x2_sol, err = solution.solve_fixed_iters()
print(f"Final solution: x_1 = {x1_sol}, x_2 = {x2_sol}, err = {err}")
# The above code is a Python implementation of the Fixed Point Iteration method for solving a system of two nonlinear equations.