batch_size = [651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]
machine_efficiency = [23, 26, 30, 34, 43, 48, 52, 57, 58]
def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum([x[i] * y[i] for i in range(n)])
    sum_x_sq = sum([x[i] ** 2 for i in range(n)])
    
 
    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x ** 2)
    a = (sum_y - b * sum_x) / n
    r = (n * sum_xy - sum_x * sum_y) / ((n * sum_x_sq - sum_x ** 2) * (n * sum([y[i] ** 2 for i in range(n)]) - sum_y ** 2)) ** 0.5
    r_sq = r ** 2
    return a, b, r, r_sq

#cuadrática
def quadratic_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_sq = sum([x[i] ** 2 for i in range(n)])
    sum_x_cube = sum([x[i] ** 3 for i in range(n)])
    sum_x_quad = sum([x[i] ** 4 for i in range(n)])
    sum_xy = sum([x[i] * y[i] for i in range(n)])
    sum_x_sq_y = sum([x[i] ** 2 * y[i] for i in range(n)])
    # Matriz de coeficientes
    A = [[n, sum_x, sum_x_sq],
         [sum_x, sum_x_sq, sum_x_cube],
         [sum_x_sq, sum_x_cube, sum_x_quad]]
    # Vector de constantes
    B = [sum_y, sum_xy, sum_x_sq_y]
    # Solución del sistema de ecuaciones
    coefficients = solve_system(A, B)
    # Coeficiente de correlación
    y_mean = sum_y / n
    total_ss = sum([(y[i] - y_mean) ** 2 for i in range(n)])
    regression_ss = coefficients[0] * n + coefficients[1] * sum_x + coefficients[2] * sum_x_sq
    r = (total_ss - regression_ss) / total_ss
    # Coeficiente de determinación
    r_sq = r ** 2
    return coefficients, r, r_sq

# Función para resolver sistemas de ecuaciones
def solve_system(A, B):
    n = len(A)
    X = [0] * n
    
    # Eliminación gaussiana
    for i in range(n):
        max_index = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[max_index][i]):
                max_index = j
        A[i], A[max_index] = A[max_index], A[i]
        B[i], B[max_index] = B[max_index], B[i]
        
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]
    for i in range(n - 1, -1, -1):
        X[i] = (B[i] - sum([A[i][j] * X[j] for j in range(i + 1, n)])) / A[i][i]
    
    return X
def cubic_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_sq = sum([x[i] ** 2 for i in range(n)])
    sum_x_cube = sum([x[i] ** 3 for i in range(n)])
    sum_x_quad = sum([x[i] ** 4 for i in range(n)])
    sum_x_quint = sum([x[i] ** 5 for i in range(n)])
    sum_x_sex = sum([x[i] ** 6 for i in range(n)])
    sum_xy = sum([x[i] * y[i] for i in range(n)])
    sum_x_sq_y = sum([x[i] ** 2 * y[i] for i in range(n)])
    sum_x_cube_y = sum([x[i] ** 3 * y[i] for i in range(n)])
    A = [[n, sum_x, sum_x_sq, sum_x_cube],
         [sum_x, sum_x_sq, sum_x_cube, sum_x_quad],
         [sum_x_sq, sum_x_cube, sum_x_quad, sum_x_quint],
         [sum_x_cube, sum_x_quad, sum_x_quint, sum_x_sex]]
    B = [sum_y, sum_xy, sum_x_sq_y, sum_x_cube_y]
    coefficients = solve_system(A, B)
    # Coeficiente de correlación
    y_mean = sum_y / n
    total_ss = sum([(y[i] - y_mean) ** 2 for i in range(n)])
    regression_ss = coefficients[0] * n + coefficients[1] * sum_x + coefficients[2] * sum_x_sq + coefficients[3] * sum_x_cube
    r = (total_ss - regression_ss) / total_ss
    # Coeficiente de determinación
    r_sq = r ** 2
    return coefficients, r, r_sq

# lineal
a_linear, b_linear, r_linear, r_sq_linear = linear_regression(batch_size, machine_efficiency)
# cuadrática
coefficients_quadratic, r_quadratic, r_sq_quadratic = quadratic_regression(batch_size, machine_efficiency)
# cúbica
coefficients_cubic, r_cubic, r_sq_cubic = cubic_regression(batch_size, machine_efficiency)


print("Regresión Lineal:")
print(f"y = {a_linear:.2f} + {b_linear:.4f} * Batch size")
print(f"Coeficiente de correlación: {r_linear:.4f}")
print(f"Coeficiente de determinación: {r_sq_linear:.4f}\n")

print("Regresión Cuadrática:")
print(f"y = {coefficients_quadratic[0]:.2f} + {coefficients_quadratic[1]:.4f} * Batch size + {coefficients_quadratic[2]:.6f} * Batch size^2")
print(f"Coeficiente de correlación: {r_quadratic:.4f}")
print(f"Coeficiente de determinación: {r_sq_quadratic:.4f}\n")

print("Regresión Cúbica:")
print(f"y = {coefficients_cubic[0]:.2f} + {coefficients_cubic[1]:.4f} * Batch size + {coefficients_cubic[2]:.5f} * Batch size^2 + {coefficients_cubic[3]:.7f} * Batch size^3")
print(f"Coeficiente de correlación: {r_cubic:.4f}")
print(f"Coeficiente de determinación: {r_sq_cubic:.4f}")