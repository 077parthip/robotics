import matplotlib.pyplot as plt

# Function to input the matrix A (step-by-step)
def input_matrix(n):
    print("Enter the matrix A (coefficients):")
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            value = float(input(f"Enter A[{i+1}][{j+1}]: "))
            row.append(value)
        A.append(row)
    return A

# Function to input the vector b
def input_vector(n):
    print("Enter the vector b (constants):")
    b = []
    for i in range(n):
        value = float(input(f"Enter b[{i+1}]: "))
        b.append(value)
    return b

# Function to perform Gaussian Elimination manually
def gaussian_elimination(A, b):
    n = len(b)

    # Forward Elimination
    for i in range(n):
        # Make the diagonal element 1 (pivot normalization if needed)
        pivot = A[i][i]
        if pivot == 0:
            raise ValueError("Zero pivot encountered!")
        for j in range(i, n):
            A[i][j] = A[i][j] / pivot
        b[i] = b[i] / pivot

        # Eliminate below rows
        for k in range(i + 1, n):
            factor = A[k][i]
            for j in range(i, n):
                A[k][j] = A[k][j] - factor * A[i][j]
            b[k] = b[k] - factor * b[i]

    # Back Substitution
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        sum_ax = 0
        for j in range(i + 1, n):
            sum_ax += A[i][j] * x[j]
        x[i] = b[i] - sum_ax
    return x

# Function to transpose a matrix manually
def transpose_matrix(A):
    rows = len(A)
    cols = len(A[0])
    T = []
    for i in range(cols):
        row = []
        for j in range(rows):
            row.append(A[j][i])
        T.append(row)
    return T

# Function to print a matrix
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:.2f}" for val in row))

# Function to visualize a 2x2 system of linear equations
def visualize_2x2_system(A, b):
    if len(A) != 2 or len(A[0]) != 2:
        print("Visualization is only supported for 2x2 systems.")
        return

    x_vals = [x / 20.0 for x in range(-200, 200)]  # From -10 to 10
    eq1_y = []
    eq2_y = []

    for x in x_vals:
        # Avoid divide by zero
        if A[0][1] != 0:
            y1 = (b[0] - A[0][0] * x) / A[0][1]
            eq1_y.append(y1)
        else:
            eq1_y.append(None)

        if A[1][1] != 0:
            y2 = (b[1] - A[1][0] * x) / A[1][1]
            eq2_y.append(y2)
        else:
            eq2_y.append(None)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, eq1_y, label='Equation 1', color='blue')
    plt.plot(x_vals, eq2_y, label='Equation 2', color='red')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title("2x2 Linear System")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

# Main function
def main():
    print("----- Linear System Solver -----")
    print("This program solves a system of linear equations using manual Gaussian Elimination.\n")

    # Step 1: Input order and values
    n = int(input("Enter the order of the square matrix (2 or 3): "))
    A = input_matrix(n)
    b = input_vector(n)

    # Step 2: Display input
    print("\nMatrix A:")
    print_matrix(A)
    print("\nVector b:")
    for i in range(n):
        print(f"b[{i+1}] = {b[i]:.2f}")

    # Step 3: Solve the system using Gaussian Elimination
    print("\nSolving using Gaussian Elimination...")
    A_copy = [row[:] for row in A]  # To avoid modifying the original matrix
    b_copy = b[:]
    solution = gaussian_elimination(A_copy, b_copy)

    print("\nSolution (x values):")
    for i in range(n):
        print(f"x{i+1} = {solution[i]:.4f}")

    # Step 4: Transpose of matrix A
    print("\nTranspose of Matrix A:")
    T = transpose_matrix(A)
    print_matrix(T)

    # Step 5: Visualize if 2x2
    if n == 2:
        print("\nGraphical Visualization of the 2x2 System:")
        visualize_2x2_system(A, b)

# Run the main function
if __name__ == "__main__":
    main()
