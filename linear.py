import numpy as np # type: ignore

import matplotlib.pyplot as plt # type: ignore

def input_matrix(n):
    matrix_elements = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if len(row) != n:
            raise ValueError(f"Row {i+1} must have {n} elements")
        matrix_elements.append(row)
    return np.array(matrix_elements)

def input_vector(n):
    vector_elements = []
    for i in range(n):
        val = float(input(f"Element {i+1}: "))
        vector_elements.append(val)
    return np.array(vector_elements)

def input_system():
    n = int(input("Order of matrix: "))
    A = input_matrix(n)
    b = input_vector(n)
    return A, b

def gauss_elimination(a, b):
    n = len(b)
    a = a.astype(float)
    b = b.astype(float)
    for i in range(n):
        for j in range(i+1, n):
            ratio = a[j][i] / a[i][i]
            a[j, i:] -= ratio * a[i, i:]
            b[j] -= ratio * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(a[i, i+1:], x[i+1:])) / a[i, i]
    return x

def inverse_matrix(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None

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

def visualize_2x2_system(A, b):
    if A.shape != (2, 2) or b.shape != (2,):
        print("2x2 systems only.")
        return

    x = np.linspace(-10, 10, 400)
    y1 = (b[0] - A[0, 0] * x) / A[0, 1]
    y2 = (b[1] - A[1, 0] * x) / A[1, 1]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, label=f'{A[0, 0]:.2f}x + {A[0, 1]:.2f}y = {b[0]:.2f}')
    plt.plot(x, y2, label=f'{A[1, 0]:.2f}x + {A[1, 1]:.2f}y = {b[1]:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Equations')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

def main():
    print("Linear Algebra Tool")
    A, b = input_system()
    print("\nGauss Elimination:")
    solution = gauss_elimination(A.copy(), b.copy())
    print("Solution:")
    for idx, val in enumerate(solution, 1):
        print(f"x{idx} = {val:.4f}")

    inv = inverse_matrix(A)
    print("\nInverse:" if inv is not None else "\nNo inverse.")
    if inv is not None:
        print(inv)

    L, U = lu_decomposition(A)
    if L is not None and U is not None:
        print("\nL matrix:")
        print(L)
        print("U matrix:")
        print(U)
    else:
        print("\nLU failed.")

    print("\nTranspose:")
    print(A.T)

    if A.shape == (2, 2):
        visualize_2x2_system(A, b)

if __name__ == "__main__":
    main()
