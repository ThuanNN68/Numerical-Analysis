import numpy as np
import sympy as sp

class Newton:
    
    def __init__(self, f, g, a, b, c, d, x_0, y_0, iter):
        x, y = sp.symbols("x y")
        self.f = sp.lambdify((x, y), f, "numpy")
        self.g = sp.lambdify((x, y), g, "numpy")
        self.grad_fx = sp.lambdify((x, y), sp.diff(f, x), 'numpy') 
        self.grad_fy = sp.lambdify((x, y), sp.diff(f, y), 'numpy') 
        self.grad_gx = sp.lambdify((x, y), sp.diff(g, x), 'numpy') 
        self.grad_gy = sp.lambdify((x, y), sp.diff(g, y), 'numpy')
 
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x_0 = x_0
        self.y_0 = y_0
        self.iter = iter
        
    def D(self, x, y):
        return self.grad_fx(x, y)*self.grad_gy(x, y)-self.grad_fy(x, y)*self.grad_gx(x, y)

    def Dx(self, x, y):
        return (-1)*self.grad_gy(x, y)*(self.f(x, y))+self.grad_fy(x, y)*self.g(x, y)
    
    def Dy(self, x, y):
        return (-1)*self.grad_fx(x, y)*(self.g(x, y))+self.grad_gx(x, y)*self.f(x, y)
        
    def solve_newton(self):
        print(f"Initialization: x_0 = {self.x_0} , y_0 = {self.y_0}")
        x_k = self.x_0
        y_k = self.y_0
        err = 0
        
        threshold = 1e-8
        
        for i in range(self.iter):
            D_val = self.D(x_k, y_k)
            if D_val < threshold:  
                print("D(x, y) approximate 0")
                break
            
            Dx_val = self.Dx(x_k, y_k)
            Dy_val = self.Dy(x_k, y_k)
            
            x_next = x_k + (Dx_val / D_val)
            y_next = y_k + (Dy_val / D_val)
            
            err = np.sqrt((x_next - x_k)**2 + (y_next - y_k)**2)
            print(f"Iter {i+1}: x = {x_next}, y = {y_next}, err = {err}")
            
            x_k = x_next
            y_k = y_next

        return x_k, y_k, err
    
# Example for test 
# x, y = sp.symbols("x y")
# f = x + 3*(sp.log(x)/sp.log(10)) - y**2
# g = 2*(x**2) -x*y - 5 *x +1

# a = 3.3
# b = 3.5
# c = 2.2
# d = 2.3
# x_0 = 3.4
# y_0 = 2.2
# iter = 20

# solver = Newton(f, g, a, b, c, d, x_0, y_0, iter)
# solver.solve_newton()