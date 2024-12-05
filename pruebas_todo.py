import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis, pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import gower


class DistanciaMahalanobis:
    def __init__(self, matriz):
        self.matriz = matriz

    def calcular_matriz_distancias(self):
        # Validación de matriz numérica
        if not np.issubdtype(self.matriz.dtype, np.number):
            raise ValueError("La matriz debe contener solo valores numéricos para Mahalanobis.")

        inv_cov_matrix = np.linalg.inv(np.cov(self.matriz.T))
        distancias = squareform(
            pdist(self.matriz, metric=lambda u, v: mahalanobis(u, v, inv_cov_matrix))
        )
        return distancias

class DistanciaGower:
    def __init__(self, matriz):
        self.matriz = matriz

    def preprocesar_matriz(self):
        """
        Convierte una matriz mixta en una matriz procesable para Gower,
        codificando las columnas categóricas.
        """
        try:
            # Convertir matriz a DataFrame
            df = pd.DataFrame(self.matriz)

            # Identificar columnas categóricas
            columnas_categoricas = df.select_dtypes(include=['object']).columns

            # Codificar las columnas categóricas
            for col in columnas_categoricas:
                df[col] = df[col].astype('category').cat.codes

            # Retornar como matriz numpy
            return df.to_numpy(dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Error al preprocesar matriz para Gower: {e}")

    def calcular_matriz_distancias(self):
        """
        Calcula la matriz de distancias de Gower después de preprocesar la matriz.
        """
        try:
            matriz_procesada = self.preprocesar_matriz()
            distancias = gower.gower_matrix(matriz_procesada)
            return distancias
        except Exception as e:
            raise ValueError(f"Error al calcular distancias Gower: {e}")

class DistanciaEsquemaBinario:
    def __init__(self, matriz):
        self.matriz = matriz

    def calcular_matriz_distancias(self):
        # Validación de matriz binaria
        if not np.isin(self.matriz, [0, 1]).all():
            raise ValueError("La matriz debe contener solo valores binarios (0 o 1) para el esquema binario.")

        distancias = squareform(pdist(self.matriz, metric='hamming'))
        return distancias

class AgrupamientoJerarquico:
    def __init__(self, matriz_distancias):
        self.matriz_distancias = matriz_distancias
        self.historial = []

    def preguntar_tipo_enlace(self):
        print("Seleccione el tipo de enlace:")
        print("1: VMC (mínimo)")
        print("2: VML (máximo)")
        print("3: Centroide")
        opcion = input("Ingrese el número correspondiente: ")
        return {"1": "single", "2": "complete", "3": "centroid"}.get(opcion, "single")

    def generar_dendograma(self):
        tipo_enlace = self.preguntar_tipo_enlace()
        print(f"Tipo de enlace seleccionado: {tipo_enlace.capitalize()}")

        # Realizar el agrupamiento
        Z = linkage(squareform(self.matriz_distancias), method=tipo_enlace)

        # Mostrar cómo se actualiza la matriz de distancias
        for i, paso in enumerate(Z):
            print(f"\nPaso {i + 1}:")
            print(f"Agrupando elementos {int(paso[0])} y {int(paso[1])} con distancia {paso[2]:.3f}")
            self.historial.append((int(paso[0]), int(paso[1]), paso[2]))

        # Graficar el dendrograma
        plt.figure(figsize=(10, 6))
        dendrogram(Z, color_threshold=0, above_threshold_color="blue")
        plt.title("Dendrograma")
        plt.show()

# Función principal para integrar todo
def main(matriz, opcion):

    try:
        if opcion == 1:
            clase_distancia = DistanciaMahalanobis(matriz)
        elif opcion == 2:
            clase_distancia = DistanciaGower(matriz)
        elif opcion == 3:
            clase_distancia = DistanciaEsquemaBinario(matriz)
        else:
            print("Opción no válida. Saliendo del programa.")
            return

        matriz_distancias = clase_distancia.calcular_matriz_distancias()
        print("\nMatriz de distancias calculada:")
        print(matriz_distancias)

        agrupador = AgrupamientoJerarquico(matriz_distancias)
        agrupador.generar_dendograma()
    except ValueError as e:
        print(f"Error: {e}")


