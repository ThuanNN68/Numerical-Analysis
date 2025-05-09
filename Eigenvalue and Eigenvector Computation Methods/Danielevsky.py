# Test the Danielevsky class with the exact matrix
A = np.array([
    [1, 2, 3, 4],
    [2, 1, 2, 3],
    [3, 2, 1, 2],
    [4, 3, 2, 1]
])

print("\n=== Testing with Exact Matrix ===")
print("Input Matrix A:")
print(np.round(A, 6))

problem = Danielevsky(A, tol=1e-10)
eigenvalues, eigenvectors = problem.solve()

print("\n=== Results ===")
print("Eigenvalues:")
print(np.round(eigenvalues, 6))
print("\nEigenvectors:")
eigenvectors_array = np.array([v.flatten() for v in eigenvectors]).T
print(np.round(eigenvectors_array, 6))

# Verify results
print("\n=== Verification ===")
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors)):
    Av = A @ vec
    lv = val * vec
    error = np.linalg.norm(Av - lv)
    print(f"\nEigenvalue/Eigenvector pair {i+1}:")
    print(f"λ = {val}")
    print(f"v = {np.round(vec.flatten(), 6)}")
    print(f"A·v = {np.round(Av.flatten(), 6)}")
    print(f"λ·v = {np.round(lv.flatten(), 6)}")
    print(f"Error |A·v - λ·v| = {error}") 