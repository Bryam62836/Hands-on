datos = {
    'Sales': [651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518],
    'Advertising': [23, 26, 30, 34, 43, 48, 52, 57, 58]
}

# o\o\o\o\o\o\o\o\oo\o\o\ooo
n = len(datos['Sales'])
sum_x = sum(datos['Advertising'])
sum_y = sum(datos['Sales'])
sum_xy = sum(x * y for x, y in zip(datos['Advertising'], datos['Sales']))
sum_x_squared = sum(x ** 2 for x in datos['Advertising'])
sum_y_squared = sum(y ** 2 for y in datos['Sales'])

# Calcular necesarios para la regresión lineal y ya 
Ex = sum_x / n
Ey = sum_y / n
Exy = sum_xy / n
Ex2 = sum_x_squared / n
Ey2 = sum_y_squared / n

# (m)
m = (Exy - Ex * Ey) / (Ex2 - Ex ** 2)

# (b)
b = Ey - m * Ex

# (coeficiente de correlación)
r_numerador = n * sum_xy - sum_x * sum_y
r_denominador = ((n * sum_x_squared - sum_x ** 2) * (n * sum_y_squared - sum_y ** 2)) ** 0.5
r = r_numerador / r_denominador

# (coeficiente de determinación)
R_squared = r ** 2

print("m:", m)
print("b:", b)
print("r:", r)
print("R^2:", R_squared)

# Ecuación de la recta de regresión
print(f"Ecuación de la recta de regresión: Y = {round(b, 2)} + {round(m, 2)}X")

# Predicciones sobre los datos 
valores_dato_predictor = [20, 25, 30, 35, 40]

for dato_predictor in valores_dato_predictor:
    print('\nValor X =', dato_predictor)
    print('Predicción =', round(b + m * dato_predictor, 4))
