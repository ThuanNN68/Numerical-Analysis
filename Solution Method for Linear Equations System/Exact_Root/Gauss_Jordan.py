import numpy as np
from sympy import symbols, simplify
import pandas as pd
import os
print("Foder")
class Gauss_Jordan():
    def __init__(self, A, B):
        self.A = A.astype(float)
        if B.ndim == 1:
            self.B = B.reshape(-1, 1)
        else:
            self.B = B
        self.p = self.B.shape[1]
        self.n = self.A.shape[0]
        self.Ab = np.hstack((self.A, self.B))
        # Initialize the solution vector
        self.x = np.zeros(self.n)
        self.solved_row = []  # List of solved rows
        self.solved_column = []  # List of solved columns
        self.epsilon = 1e-8
    
    def is_integer_not_0(self, value):
        return ((abs(value - round(value)) < self.epsilon) and (abs(value) > self.epsilon))
    
    def find_pivot(self):
        max_row = None
        max_column = None
        max_value = self.epsilon
        
        for row in range(0, self.n):
            if row in self.solved_row:
                continue

            for col in range(0, self.n):
                if col in self.solved_column:
                    continue
                if self.is_integer_not_0(self.A[row, col]):
                    return row, col
                if abs(self.A[row, col]) > self.epsilon:
                    max_value = abs(self.A[row, col])
                    max_row = row
                    max_column = col
        if max_row is None or max_column is None:
            return None, None
        return max_row, max_column

    def solve(self):
        for i in range(self.n):
            row_pivot, col_pivot = self.find_pivot()
            if row_pivot is None or col_pivot is None:
                print("No more pivots available.")
                break
            print(f"Pivot element at row {row_pivot}, col {col_pivot}: {self.A[row_pivot, col_pivot]}")
            
            for row in range(self.n):
                if row != row_pivot:
                    self.Ab[row] -= self.Ab[row_pivot] * (self.Ab[row, col_pivot] / self.A[row_pivot, col_pivot])
                    self.A[row] -= self.A[row_pivot] * (self.A[row, col_pivot] / self.A[row_pivot, col_pivot])
                    self.Ab[row, col_pivot] = 0.0
        
            # Mark the row and column as solved
            self.solved_row.append(row_pivot)
            self.solved_column.append(col_pivot)
            print(np.array2string(self.Ab, formatter={'float_kind': lambda x: "%.5f" % x}))
        
        def is_zero_row(row):
            return np.all(np.abs(row[:self.n]) < self.epsilon) and np.all(np.abs(row[self.n:]) < self.epsilon)
        def is_inconsistent_row(r):
            return np.all(np.abs(r[:self.n]) < self.epsilon) and np.any(np.abs(r[self.n:]) > self.epsilon)
    
        rankAb = sum(not is_zero_row(self.Ab[i, :]) for i in range(self.n))
        rankA = len(self.solved_row)
        
        # Check if the system is consistent
        if any(is_inconsistent_row(self.Ab[i, :]) for i in range(self.n)):
            print("The system is inconsistent.")
            return None
        
        elif(rankA < self.n):
            print("The system has infinitely many solutions.")
            # Check for free variables
            pivot_columns = self.solved_column
            print(f"Pivot columns: {pivot_columns}")

            all_vars = set(range(self.n))
            free_vars = list(all_vars - set(pivot_columns))
            free_syms = {j: symbols(f"t{j}") for j in free_vars}
            print(f"Free variables: {free_vars}")
            print(f"Free symbols: {free_syms}")

            # Create symbolic solution for each column in B
            solutions = []
            for col_b in range(self.p):
                # Initialize solution dictionary for current B column
                solution_dict = {}
                
                # For pivot variables
                for i, col in enumerate(pivot_columns):
                    row = self.solved_row[i]
                    # Start with the constant term from B
                    value = self.Ab[row, self.n + col_b] / self.Ab[row, col]
                    
                    # Subtract contributions from free variables
                    expr = value
                    for free_var in free_vars:
                        coef = -self.Ab[row, free_var] / self.Ab[row, col]
                        if abs(coef) > self.epsilon:
                            expr += coef * free_syms[free_var]
                    
                    solution_dict[col] = simplify(expr)
                
                # For free variables (they equal their symbols)
                for j in free_vars:
                    solution_dict[j] = free_syms[j]
                
                solutions.append(solution_dict)
            return solutions
            
     
        else:
            print("The system has a unique solution.")
            for i in range(self.n):
                for col in range(self.p):
                    self.x[i] = self.Ab[i, self.n + col] / self.Ab[i, i]
            return self.x
        
# Example usage
# A = np.array([[1, 1, 1], [2, 3, 5], [4, 0, 5]])
# b = np.array([3, 10, 9])
df = pd.read_excel("Numerical-Analysis\Solution Method for Linear Equations System\Exact_Root\gts1.xlsx", sheet_name=0, engine='openpyxl')
arr = df.to_numpy()
A = arr[:,:-2]
B = arr[:,-2:]
solution = Gauss_Jordan(A, B)
sol = solution.solve()

# Check if the solution is a list (indicating infinite solutions) or a numpy array (indicating a unique solution)
if sol is not None:
    # Check if the solution is a list (indicating infinite solutions)
    if isinstance(sol, list): 
        for idx, s in enumerate(sol):
            print(f"\n {idx + 1}:")
            for var, expr in sorted(s.items()):
                print(f"{var} = {expr}")
    else:
        print("Solution X:")
        print(sol)
else:
    print("No solution found.")
