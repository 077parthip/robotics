import numpy as np
import matplotlib.pyplot as plt

# Step 1: Input Functions
def input_matrix(n):
    print(f"\nEnter {n}x{n} matrix A row by row:")
    A = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        while len(row) != n:
            print("Invalid! Enter exactly", n, "values.")
            row = list(map(float, input(f"Row {i+1}: ").split()))
        A.append(row)
    return np.array(A)

def input_vector(n):
    print(f"\nEnter {n} values for vector b:")
    b = []
    for i in range(n):
        val = float(input(f"b[{i+1}]: "))
        b.append(val)
    return np.array(b)

# Step 2: Gaussian Elimination
def gaussian_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    for i in range(n):
        for j in range(i+1, n):
            ratio = A[j][i] / A[i][i]
            A[j, i:] -= ratio * A[i, i:]
            b[j] -= ratio * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i][i]
    return x

# Step 3: Inverse Method
def inverse_method(A, b):
    try:
        A_inv = np.linalg.inv(A)
        x = A_inv @ b
        return x, A_inv
    except np.linalg.LinAlgError:
        return None, None

# Step 4: LU Decomposition
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i+1, n):
            if U[i][i] == 0:
                return None, None
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    return L, U

# Step 5: Transpose
def print_transpose(A):
    print("\n🔹 Transpose of A:")
    print(A.T)

# Step 6: Plot (only for 2x2)
def plot_2x2_system(A, b):
    if A.shape != (2, 2):
        print("\nSkipping visualization (only for 2x2).")
        return
    x_vals = np.linspace(-10, 10, 400)
    try:
        y1 = (b[0] - A[0][0] * x_vals) / A[0][1]
        y2 = (b[1] - A[1][0] * x_vals) / A[1][1]
    except ZeroDivisionError:
        print("Cannot plot — division by zero in equations.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y1, label='Equation 1', color='blue')
    plt.plot(x_vals, y2, label='Equation 2', color='red')
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Graphical Representation of 2x2 System")
    plt.legend()
    plt.show()

# Step 7: Main Driver
def main():
    print("==== Linear System Solver ====")
    n = int(input("Enter matrix size (2 or 3): "))
    if n not in [2, 3]:
        print("Only 2x2 or 3x3 systems supported.")
        return

    A = input_matrix(n)
    b = input_vector(n)

    print("\n🔹 Gaussian Elimination:")
    x_gauss = gaussian_elimination(A.copy(), b.copy())
    for i in range(n):
        print(f"x{i+1} = {x_gauss[i]:.4f}")

    print("\n🔹 Inverse Matrix Method:")
    x_inv, A_inv = inverse_method(A, b)
    if x_inv is not None:
        for i in range(n):
            print(f"x{i+1} = {x_inv[i]:.4f}")
        print("\nInverse of A:")
        print(A_inv)
    else:
        print("Matrix is singular — no inverse.")

    print("\n🔹 LU Decomposition:")
    L, U = lu_decomposition(A)
    if L is not None:
        print("L matrix:")
        print(L)
        print("U matrix:")
        print(U)
    else:
        print("LU decomposition failed (zero pivot).")

    print_transpose(A)

    if n == 2:
        plot_2x2_system(A, b)

if __name__ == "__main__":
    main()
