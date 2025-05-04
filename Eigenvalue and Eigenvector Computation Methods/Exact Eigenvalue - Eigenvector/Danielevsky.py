import numpy as np

class Danielevsky:
    def __init__(self, A, tol=1e-10):
        self.A = A
        self.n = len(A[0])
        self.tol = tol

    def find_similarity_matrix(self, A, k):
        n = len(A[0])
        M = np.eye(n)
        for j in range(n):
            if j == k - 1:
                M[k-1, j] = 1 / A[k, k-1]
            else:
                M[k-1, j] = -A[k, j] / A[k, k-1]
        inverse_M = np.eye(n)
        for j in range(n):
            inverse_M[k-1, j] = A[k, j]
        return M, inverse_M

    def find_eigenvalues_vectors(self):
        if self.n == 0:
            return [], []
        if self.n == 1:
            return [self.A[0, 0]], [np.array([[1.0]])]

        A = self.A.copy()
        back_transform = np.eye(self.n)
        eigenvalues = []
        eigenvectors = []

        m = self.n
        k = self.n - 1

        while k > 0:
            pivot = A[k, k-1]
            if abs(pivot) > self.tol:
                M, M_inv = self.find_similarity_matrix(A, k)
                A = M @ A @ M_inv
                back_transform = back_transform @ M_inv
                k -= 1
            else:
                # Find a non-zero element in the row
                found = False
                for j in range(k-1):
                    if abs(A[k, j]) > self.tol:
                        P = np.eye(self.n)
                        # Swap rows of identity matrix to bring a non-zero element to the pivot position
                        P[:, [j, k-1]] = P[:, [k-1, j]]
                        # Tranform the matrix A
                        A = P @ A @ P
                        back_transform = back_transform @ P
                        found = True
                        break
                if not found:
                    # Deflation
                    X_sub = A[k:self.n, k:self.n]
                    t = X_sub.shape[0]
                    if t > 0:
                        # Eigenvalue computation for the submatrix
                        sub_eigenvalues, sub_eigenvectors = np.linalg.eig(X_sub)
                        for i in range(t):
                            val = sub_eigenvalues[i]
                            vec_sub = sub_eigenvectors[:, i:i+1]
                            eigenvalues.append(val)
                            full_vec = np.zeros((self.n, 1), dtype = float)
                            full_vec[k:self.n, 0] = vec_sub.flatten()
                            v = back_transform @ full_vec
                            eigenvectors.append(v)
                    m = k
                    k -= 1

        if m > 0:
            final_block = A[0:m, 0:m]
            t = final_block.shape[0]
            roots = np.linalg.eigvals(final_block)

            for val in roots:
                eigenvalues.append(val)
                y = np.power(val, np.arange(t-1, -1, -1)).reshape((t, 1))
                full_vec = np.zeros((self.n, 1), dtype= float)
                full_vec[0:t, 0] = y.flatten()
                v = back_transform @ full_vec
                eigenvectors.append(v)
        return eigenvalues, eigenvectors

    def check_eigenvalues(self, eigenvalues, eigenvectors):
        print("\n--- Checking Eigenvalues/Eigenvectors ---")
        correct = True
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors)):
            Av = self.A @ vec
            lv = val * vec
            error = np.linalg.norm(Av - lv)
            if error > 1e-3:
                print(f"Error: Eigenvector {i+1} is incorrect.")
                correct = False
        if correct:
            print("All eigenvalue/eigenvector pairs seem correct within tolerance.")
        else:
            print("Potential issues found in eigenvalue/eigenvector pairs.")
        return correct

    def solve(self):
        print("--- Danielevsky Method ---")
        print("Initial Matrix A:\n", np.round(self.A, 6))
        eigenvalues, eigenvectors = self.find_eigenvalues_vectors()
        print("Check Eigenvalues/Eigenvectors:")
        self.check_eigenvalues(eigenvalues, eigenvectors)
        print("\nCalculated Eigenvalues and Eigenvectors (Danielevsky):")
        if not eigenvalues:
            print("No eigenvalues found.")
        for i, (value, vector) in enumerate(zip(eigenvalues, eigenvectors)):
            print(f"Eigenvalue {i+1}: {np.round(value, 6)}")
            if vector is not None:
                print(f"Eigenvector {i+1}:\n{np.round(vector.flatten(), 6)}")
            else:
                print(f"Eigenvector {i+1}: None")

        print("\nEigenvalues and Eigenvectors comparison with numpy:")
        np_eigenvalues, np_eigenvectors = np.linalg.eig(self.A)
        for i, (value, vector) in enumerate(zip(np_eigenvalues, np_eigenvectors.T)):
            print(f"Eigenvalue {i+1}: {np.round(value, 6)}")
            if vector is not None:
                print(f"Eigenvector {i+1}:\n{np.round(vector.flatten(), 6)}")
            else:
                print(f"Eigenvector {i+1}: None")
        return eigenvalues, eigenvectors

# Test the Danielevsky class with a sample matrix
A = np.array([
    [1, 2, 0, 0],
    [3, 4, 0, 0],
    [0, 0, 5, 6],
    [0, 0, 7, 8]
])

tol = 1e-10
problem = Danielevsky(A, tol)
eigenvalues, eigenvectors = problem.solve()
