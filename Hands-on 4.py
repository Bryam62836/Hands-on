from collections import Counter
import math

class KNN:
    def _init_(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return predictions

    def _predict(self, x):
        # Calcular la distancia entre x y todos los puntos de entrenamiento
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        # Obtener los índices de los k puntos más cercanos
        k_indices = self._argsort(distances)[:self.k]
        # Obtener las etiquetas de los k puntos más cercanos
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Devolver la etiqueta más común
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

    def _argsort(self, seq):
        return sorted(range(len(seq)), key=seq._getitem_)

# Ejemplo de uso
if _name_ == "_main_":
    # Datos de entrenamiento
    X_train = [[1, 2], [2, 3], [3, 4], [6, 5], [7, 8]]
    y_train = [0, 0, 0, 1, 1]

    # Datos de prueba
    X_test = [[1, 1], [5, 5]]

    # Crear el clasificador kNN
    k = 3
    knn = KNN(k=k)

    # Entrenar el clasificador
    knn.fit(X_train, y_train)

    # Predecir las etiquetas para los datos de prueba
    predictions = knn.predict(X_test)

    print(f"Predicciones para X_test: {predictions}")