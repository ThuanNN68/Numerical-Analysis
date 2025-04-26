import numpy as np
from sympy import symbols, simplify, sympify
import pandas as pd

class Gauss():
    def __init__(self, A, B):
        self.A = A.astype(float)
        self.n = A.shape[0]
        if B.ndim == 1:
            self.B = B.reshape(-1, 1)
        else:
            self.B = B
        self.p = self.B.shape[1]  
        self.x = np.zeros((self.n, self.p))
        self.Ab = np.hstack((self.A, self.B))  
        self.epsilon = 1e-10

    def pivot(self, i):
        if abs(self.Ab[i, i]) >= self.epsilon:
            return True
        for l in range(i + 1, self.n):
            if abs(self.Ab[l, i]) > self.epsilon:
                self.Ab[[i, l],:] = self.Ab[[l, i],:]
                return True
        return False

    def solve(self):
        row = 0
        pivot_columns = [] # list to store pivot columns
        pivot_rows = []    # list to store corresponding pivot rows

        # Forward elimination
        for col in range(self.n):
            if row >= self.n:
                break
            if not self.pivot(row):
                continue
            pivot_columns.append(col)
            pivot_rows.append(row)
            pivot_element = self.Ab[row, col]
            print(f"Pivot element at row {row}, col {col}: {pivot_element}")
            for k in range(row + 1, self.n):
                if abs(self.Ab[k, col]) > self.epsilon:
                    factor = self.Ab[k, col] / pivot_element
                    self.Ab[k, col:] -= factor * self.Ab[row, col:]
                    self.Ab[k, col] = 0.0
            print(np.array2string(self.Ab, formatter={'float_kind': lambda x: "%.10f" % x}))
            row += 1
        
        print("After forward elimination:")
        print(np.array2string(self.Ab, formatter={'float_kind': lambda x: "%.10f" % x}))

        rankA = len(pivot_columns)
        
        def is_zero_row(row_arr):
            return np.all(np.abs(row_arr[:self.n]) < self.epsilon) and np.all(np.abs(row_arr[self.n:]) < self.epsilon)
            
        def is_inconsistent_row(row_arr):
            return np.all(np.abs(row_arr[:self.n]) < self.epsilon) and np.any(np.abs(row_arr[self.n:]) > self.epsilon)
            
        # Check for inconsistent system
        if any(is_inconsistent_row(self.Ab[i, :]) for i in range(self.n)):
            print("The system is inconsistent.")
            return None
        
        # Check for infinite solutions
        if rankA < self.n:
            print("The system has infinite solutions.")
            # Check for free variables
            all_vars = set(range(self.n))
            free_vars = list(all_vars - set(pivot_columns))
            free_syms = {j: symbols(f"t{j}") for j in free_vars}
            print(f"Pivot columns: {pivot_columns}")
            print(f"Free variables: {free_vars}")
            print(f"Free symbols: {free_syms}")

            solutions = []
            for b_col in range(self.p):
                sol = {}
                
                for j in free_vars:
                    sol[f"x{j}"] = free_syms[j]
                
                for i in range(rankA-1, -1, -1):
                    pivot_row = pivot_rows[i]
                    pivot_col = pivot_columns[i]
                    
                    expr = sympify(self.Ab[pivot_row, self.n + b_col])
                    
                    for j in range(pivot_col+1, self.n):
                        coef = self.Ab[pivot_row, j]
                        if abs(coef) > self.epsilon:
                            if j in free_vars:
                                expr -= coef * free_syms[j]
                            elif j in pivot_columns:
                                idx = pivot_columns.index(j)
                                pivot_j_row = pivot_rows[idx]
                                if idx > i:  
                                    expr -= coef * sol[f"x{j}"]
                    
                    expr = expr / self.Ab[pivot_row, pivot_col]
                    sol[f"x{pivot_col}"] = simplify(expr)
                
                solutions.append(sol)
            return solutions

        # Back substitution if the system has a unique solution
        for col in range(self.p):
            for i in range(rankA-1, -1, -1):
                pivot_row = pivot_rows[i]
                pivot_col = pivot_columns[i]
                sum_ax = 0
                for j in range(pivot_col+1, self.n):
                    sum_ax += self.Ab[pivot_row, j] * self.x[j, col]
                self.x[pivot_col, col] = (self.Ab[pivot_row, self.n + col] - sum_ax) / self.Ab[pivot_row, pivot_col]
        
        print(np.array2string(self.x, formatter={'float_kind': lambda x: "%.10f" % x}))
        return self.x
    
df = pd.read_excel("Numerical-Analysis/Solution Method for Linear Equations System/Exact_Root/gts1.xlsx", sheet_name=0, engine='openpyxl')
arr = df.to_numpy()
A = arr[:,:-2]
B = arr[:,-2:]
gauss = Gauss(A, B)
sol = gauss.solve()

if sol is not None:
    if isinstance(sol, list): 
        print("System has infinite solutions:")
        for idx, s in enumerate(sol):
            print(f"\nSolution for B column {idx + 1}:")
            for var, expr in sorted(s.items()):
                print(f"{var} = {expr}")
    else:
        print("System has unique solution.")
        print("Solution X:")
        print(sol)
else:
    print("No solution found.")