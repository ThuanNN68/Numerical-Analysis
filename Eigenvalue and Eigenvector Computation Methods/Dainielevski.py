import numpy as np

class Danielevsky:
    def __init__(self, A, tol=1e-5):
        self.A = A
        self.n = len(A[0])
        self.tol = tol

    def find_similarity_matrix(self, A, k):
        n = len(A[0])
        S = np.eye(n)
        S[k-1, :] = A[k, :]
        
        Sinv = np.eye(n)
        Sinv[k-1, :] = -A[k, :] / A[k, k-1]
        Sinv[k-1, k-1] = 1 / A[k, k-1]
        
        A1 = S @ A @ Sinv
        return A1, S, Sinv
    
    def characteristic_polynomial(self, A):
        n = A.shape[0]
        if n == 0:
            return A[1]
        # For a matrix in Frobenius form, the characteristic polynomial
        p = (-1)**n * np.ones(n+1)
        p[1:] = p[1: ] * (-1) *A[0, :]
        return p
    
    def findvalue(self, A_frobenius): 
        if A_frobenius.shape[0] == 0:
            return np.array([])
        p = self.characteristic_polynomial(A_frobenius)
        # np.roots expects coefficients from highest power to lowest
        eigenvalues = np.roots(p)
        return sorted(eigenvalues)
    
    def find_eigenvalues_vectors(self):
        """Main algorithm to find eigenvalues and eigenvectors using Danielevsky method"""
        # Handle special cases
        if self.n == 0:
            return [], []
        if self.n == 1:
            return [self.A[0, 0]], [np.array([[1.0]])]

        A = self.A.copy()
        # Initialize the back transformation matrix
        back_transform = np.eye(self.n)
        eigenvalues = []
        eigenvectors = []

        m = self.n
        k = self.n - 1

        while k > 0:
            pivot = A[k, k-1]
            if abs(pivot) > self.tol:
                A, S, Sinv = self.find_similarity_matrix(A, k)
                print(f"S matrix (k={k}):")
                print(np.round(S, 6))
                print(f"S^(-1) matrix (k={k}):")
                print(np.round(Sinv, 6))
                back_transform = back_transform @ Sinv
                # print(f"Back transformation matrix after update (k={k})", np.round(back_transform, 6))
                k -= 1
            
            else:
                print(f"Pivot A[{k},{k-1}] = {pivot} is zero element")
                # Find a non-zero element in the row
                found = False
                for j in range(k-1):
                    if abs(A[k, j]) > self.tol:
                        print(f"Found non-zero element A[{k}][{j}] = {A[k, j]}")
                        P = np.eye(self.n)
                        # Swap columns of identity matrix to bring a non-zero element to the pivot position
                        P[:, [j, k-1]] = P[:, [k-1, j]]
                        # print(f"Permutation matrix P for swapping columns {j} and {k-1}", P)
                        # Tranform the matrix A
                        A = P @ A @ P
                        print(f"Matrix A after permutation (k={k})", np.round(A, 6))
                        back_transform = back_transform @ P
                        # print(f"Back transformation matrix after update (k={k})", np.round(back_transform, 6))
                        found = True
                        break
                    
                if not found:
                    print(f"No non-zero element found in row {k}")
                    # Deflation
                    n = A.shape[0]
                    r = n - k
                    E_r = np.eye(r)
                    E_k = np.eye(k)
                    
                    # Extract blocks
                    B = A[k:, :k]  
                    
                    B1 = np.zeros((r, k)) 
                    B1[:, 1] = -B[:, 1]  
                    
                    B1_inv = -B1  
                    
                    S = np.block([
                        [E_r, B1],
                        [np.zeros((k, r)), E_k]
                    ])
                    
                    S_inv = np.block([
                        [E_r, -B1],
                        [np.zeros((k, r)), E_k]
                    ])

                    # Calculate A' = S A S⁻¹
                    A_prime = S @ A @ S_inv
                    back_transform = back_transform @ S_inv 
                    
                    # Additional transformations for columns 2 to k-1
                    for col in range(2, k): 
                        B_q = np.zeros((r, k)) 
                        B_q[:, 1] = -B_q[:, 1]  
                        
                        # Get column from B
                        b_column = B[:, col]
                        
                        # Create transformation matrix S_q
                        for i in range(r):  # Iterate over rows in B
                            B_q[i, col] = -b_column[i]  
                        S_q = np.block([
                            [E_r, B_q],
                            [np.zeros((k, r)), E_k]
                        ])
                        
                        # Create inverse transformation
                        S_q_inv = np.linalg.inv(S_q)
                        
                        # Apply transformation
                        A_prime = S_q @ A_prime @ S_q_inv
                        back_transform = back_transform @ S_q_inv
                    
                    # Check if the final column is zero
                    zero_final_col = True
                    for i in range(r):
                        if A_prime[i, k - 1] != 0:
                            zero_final_col = False
                            break
                    if not zero_final_col:
                        print(f"Final column is not zero")

                    A = A_prime
                    print(A)
                    X_sub = A[k:, k:]
                    t = X_sub.shape[0]
                    
                    if t > 0:
                        print(f"Computing eigenvalues and eigenvectors for submatrix")
                        sub_eigenvalues = self.findvalue(X_sub)
                        print(f"Frobenius block: {np.round(X_sub, 6)}")
                        print(f"Eigenvalues of block", np.round(sub_eigenvalues, 6))

                        for j in range(len(sub_eigenvalues)):
                            eigenvalues.append(sub_eigenvalues[j])
                            y = np.power(sub_eigenvalues[j], np.arange(t))[::-1].reshape((t, 1))
                            v = np.zeros((self.n, 1))
                            v[k:m] = y
                            eigenvectors.append(v)
                            print(f"Eigenvector after back transformation", np.round(v, 6))
                        m = k
                        A = A[:k, :k]
                k -= 1
            
        final_block = A
        t = final_block.shape[0]
        eigenvalue = self.findvalue(final_block)

        print(f"Processing final block of size {t}x{t}", np.round(final_block, 6))
        print(f"Final block matrix:\n{np.round(final_block, 6)}")
        print(f"Eigenvalues of final block:\n{np.round(eigenvalue, 6)}")

        for j in range(len(eigenvalue)):
            eigenvalues.append(eigenvalue[j])
            # For Frobenius form, eigenvector components are powers of eigenvalue
            y = np.power(eigenvalue[j], np.arange(t))[::-1].reshape((t, 1))
            v = np.zeros((self.n, 1))
            v[0:t] = y  # For final block, we start from index 0
            print(f"Eigenvector before transformation:\n{np.round(v, 6)}")
            eigenvectors.append(v)

        # Apply back transformation and normalize eigenvectors
        for i in range(len(eigenvectors)):
            v_transformed = back_transform @ eigenvectors[i]
            v_normalized = v_transformed / np.linalg.norm(v_transformed)
            eigenvectors[i] = v_normalized
            print(f"Eigenvector after transformation and normalization:\n{np.round(eigenvectors[i], 6)}")

        return eigenvalues, eigenvectors
    
    def check_eigenvalues(self, eigenvalues, eigenvectors):
        print("\n--- Checking Eigenvalues/Eigenvectors ---")
        correct = True
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors)):
            Av = self.A @ vec
            lv = val * vec
            error = np.linalg.norm(Av - lv)
            print(f"Checking pair {i+1}:")
            print(f"λ = {val}")
            print(f"v = {np.round(vec.flatten(), 6)}")
            print(f"A·v = {np.round(Av.flatten(), 6)}")
            print(f"λ·v = {np.round(lv.flatten(), 6)}")
            print(f"Error |A·v - λ·v| = {error}")
            
            if error > 1e-3:
                print(f"ERROR: Eigenvector {i+1} is incorrect (error > 1e-3)")
                correct = False
        return correct

    def solve(self):
        """Main solver method"""
        print("--- Danielevsky Method ---")
        print("Initial Matrix A:\n", np.round(self.A, 6))
        
        eigenvalues, eigenvectors = self.find_eigenvalues_vectors()
        
        print("\nCalculated Eigenvalues:")
        print(np.round(eigenvalues, 6))
        print("\nCalculated Eigenvectors:")
        eigenvectors_array = np.array([v.flatten() for v in eigenvectors]).T
        print(np.round(eigenvectors_array, 6))
        
        # Compare with NumPy
        print("\n=== Comparison with NumPy ===")
        np_eigenvalues, np_eigenvectors = np.linalg.eig(self.A)
        print("\nNumPy Eigenvalues:")
        print(np.round(np_eigenvalues, 6))
        print("\nNumPy Eigenvectors:")
        print(np.round(np_eigenvectors, 6))
        check = self.check_eigenvalues(eigenvalues, eigenvectors)
        if check:
            print("\nAll eigenvalue/eigenvector pairs are correct within tolerance.")
        else:
            print("\nPotential issues found in eigenvalue/eigenvector pairs.")
        return eigenvalues, eigenvectors

A = np.array([[1, 6, 3], [4, 2, 6], [7, 8, 9]])
problem = Danielevsky(A)
problem.solve()

