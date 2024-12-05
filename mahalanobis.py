import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

class ClusterEslabonamiento:

    def __init__(self, matriz):
        self.matriz = self.validar_matriz(matriz)  # Validar la matriz de entrada
        self.historial_eslabonamientos = []  # Historial de combinaciones de clusters

    def validar_matriz(self, matriz):
        """
        Valida que todos los elementos de la matriz sean numéricos.
        Convierte cadenas que representen números en flotantes.
        """
        matriz_validada = []
        for fila in matriz:
            nueva_fila = []
            for elemento in fila:
                try:
                    nueva_fila.append(float(elemento))  # Convertir a número
                except ValueError:
                    raise ValueError(f"Elemento no numérico encontrado: {elemento}")
            matriz_validada.append(nueva_fila)
        return matriz_validada

    def mostrar_matriz(self, nombre, matriz, decimales=3):
        """
        Imprime una matriz con formato amigable y valores redondeados.
        """
        print(f"\n{nombre}:")
        for fila in matriz:
            print("  ".join(f"{round(valor, decimales):>{8}}" for valor in fila))

    def calcular_media(self, matriz):
        filas = len(matriz)
        columnas = len(matriz[0])
        media = [0] * columnas

        for col in range(columnas):
            suma = sum(fila[col] for fila in matriz)
            media[col] = suma / filas
        return media

    def centrar_matriz(self, matriz, media):
        matriz_centrada = []
        for fila in matriz:
            matriz_centrada.append([fila[i] - media[i] for i in range(len(fila))])
        return matriz_centrada

    def eliminar_columnas_constantes(self, matriz):
        columnas_a_eliminar = []
        columnas = len(matriz[0])

        for col in range(columnas):
            valores_columna = [fila[col] for fila in matriz]
            if max(valores_columna) == min(valores_columna):
                columnas_a_eliminar.append(col)

        matriz_filtrada = []
        for fila in matriz:
            nueva_fila = [valor for i, valor in enumerate(fila) if i not in columnas_a_eliminar]
            matriz_filtrada.append(nueva_fila)

        return matriz_filtrada

    def calcular_covarianza(self, matriz_centrada):
        filas = len(matriz_centrada)
        columnas = len(matriz_centrada[0])
        covarianza = [[0] * columnas for _ in range(columnas)]

        for i in range(columnas):
            for j in range(columnas):
                suma = sum(matriz_centrada[k][i] * matriz_centrada[k][j] for k in range(filas))
                covarianza[i][j] = suma / (filas - 1)
        return covarianza

    def calcular_inversa_covarianza(self, matriz_cov):
        try:
            matriz_cov_inv = np.linalg.inv(matriz_cov)
            return matriz_cov_inv
        except np.linalg.LinAlgError:
            print("Error: La matriz de covarianza es singular y no se puede invertir.")
            return None

    def calcular_distancia_mahalanobis_manual(self, vector, media, cov_inversa):
        resta = [vector[i] - media[i] for i in range(len(vector))]
        resta = np.array(resta)
        distancia = np.sqrt(resta.T @ cov_inversa @ resta)
        return distancia

    def elegir_eslabonamiento(self, matriz_distancias):
        print("\nElige el tipo de eslabonamiento:")
        print("1. Vecino más cercano")
        print("2. Vecino más lejano")
        print("3. Centroide")

        tipo = input("Introduce el número correspondiente: ").strip()

        if tipo == "1":
            metodo_dendrograma = 'single'
            self.eslabonamiento_vecino_mas_cercano(matriz_distancias)
        elif tipo == "2":
            metodo_dendrograma = 'complete'
            self.eslabonamiento_vecino_mas_lejano(matriz_distancias)
        elif tipo == "3":
            metodo_dendrograma = 'centroid'
            self.eslabonamiento_centroide(matriz_distancias)
        else:
            print("Opción no válida. Intenta de nuevo.")
            return self.elegir_eslabonamiento(matriz_distancias)

        # Generar un solo dendrograma después de calcular el eslabonamiento
        self.generar_dendrograma(matriz_distancias, metodo=metodo_dendrograma)

    def eslabonamiento_vecino_mas_cercano(self, matriz_distancias):
        self._eslabonamiento_generico(matriz_distancias, min, "Vecino más cercano")

    def eslabonamiento_vecino_mas_lejano(self, matriz_distancias):
        self._eslabonamiento_generico(matriz_distancias, max, "Vecino más lejano")

    def eslabonamiento_centroide(self, matriz_distancias):
        n = len(matriz_distancias)
        clusters = [[i + 1] for i in range(n)]

        def calcular_centroide(cluster):
            centroide = []
            for j in range(len(self.matriz[0])):
                valores = []
                for idx in cluster:
                    valor = self.matriz[idx - 1][j]
                    try:
                        valores.append(float(valor))  # Intentamos convertir a número
                    except ValueError:
                        pass  # Ignoramos valores no numéricos
                if valores:  # Si hay valores numéricos, calculamos la media
                    centroide.append(sum(valores) / len(valores))
                else:  # Si no hay valores numéricos, agregamos un marcador (por ejemplo, 0)
                    centroide.append(0)
            return centroide

        while len(clusters) > 1:
            min_dist = float('inf')
            clust1, clust2 = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    centroide_1 = calcular_centroide(clusters[i])
                    centroide_2 = calcular_centroide(clusters[j])

                    distancia = np.sqrt(sum((x - y) ** 2 for x, y in zip(centroide_1, centroide_2)))
                    if distancia < min_dist:
                        min_dist = distancia
                        clust1, clust2 = i, j

            nuevo_cluster = clusters[clust1] + clusters[clust2]
            self.historial_eslabonamientos.append(
                f"Combinar elemento {clusters[clust1]} y elemento {clusters[clust2]} con distancia {min_dist:.2f}\n"
            )
            clusters = [clusters[k] for k in range(len(clusters)) if k != clust1 and k != clust2]
            clusters.append(nuevo_cluster)

        print("\nHistorial de eslabonamientos:\n")
        for eslabonamiento in self.historial_eslabonamientos:
            print(eslabonamiento)

    def _eslabonamiento_generico(self, matriz_distancias, func, tipo_eslabonamiento):
        n = len(matriz_distancias)
        clusters = [[i + 1] for i in range(n)]

        while len(clusters) > 1:
            extrema_dist = float('inf') if func == min else -float('inf')
            clust1, clust2 = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    for a in clusters[i]:
                        for b in clusters[j]:
                            distancia = matriz_distancias[a - 1][b - 1]
                            if func(distancia, extrema_dist) == distancia:
                                extrema_dist = distancia
                                clust1, clust2 = i, j

            nuevo_cluster = clusters[clust1] + clusters[clust2]
            self.historial_eslabonamientos.append(
                f"Combinar elemento {clusters[clust1]} y elemento {clusters[clust2]} con distancia {extrema_dist:.2f}\n"
            )
            clusters = [clusters[k] for k in range(len(clusters)) if k != clust1 and k != clust2]
            clusters.append(nuevo_cluster)

        print(f"\nHistorial de eslabonamientos ({tipo_eslabonamiento}):\n")
        for eslabonamiento in self.historial_eslabonamientos:
            print(eslabonamiento)

    def generar_dendrograma(self, matriz_distancias, metodo):
        if metodo == 'single':
            linkage_matrix = self._generar_matriz_linkage(matriz_distancias, min)
        elif metodo == 'complete':
            linkage_matrix = self._generar_matriz_linkage(matriz_distancias, max)
        elif metodo == 'centroid':
            linkage_matrix = self._generar_matriz_linkage_centroide(matriz_distancias)
        else:
            raise ValueError("Método no válido para el dendrograma.")

        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix, labels=[f'Individuo {i + 1}' for i in range(len(self.matriz))])
        plt.title(f'Dendrograma - Método {metodo.capitalize()}')
        plt.xlabel('Individuos')
        plt.ylabel('Distancia')
        plt.show()

    def _generar_matriz_linkage(self, matriz_distancias, func):
        n = len(matriz_distancias)
        clusters = [[i] for i in range(n)]
        linkage_matrix = []
        cluster_ids = {i: i for i in range(n)}
        current_cluster = n

        while len(clusters) > 1:
            extrema_dist = float('inf') if func == min else -float('inf')
            clust1, clust2 = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    for a in clusters[i]:
                        for b in clusters[j]:
                            distancia = matriz_distancias[a][b]
                            if func(distancia, extrema_dist) == distancia:
                                extrema_dist = distancia
                                clust1, clust2 = i, j

            linkage_matrix.append([cluster_ids[clusters[clust1][0]],
                                   cluster_ids[clusters[clust2][0]],
                                   extrema_dist,
                                   len(clusters[clust1]) + len(clusters[clust2])])

            nuevo_cluster = clusters[clust1] + clusters[clust2]
            for idx in nuevo_cluster:
                cluster_ids[idx] = current_cluster
            clusters = [clusters[k] for k in range(len(clusters)) if k != clust1 and k != clust2]
            clusters.append(nuevo_cluster)
            current_cluster += 1

        return np.array(linkage_matrix)

    def _generar_matriz_linkage_centroide(self, matriz_distancias):
        n = len(matriz_distancias)
        clusters = [[i] for i in range(n)]
        linkage_matrix = []
        cluster_ids = {i: i for i in range(n)}
        current_cluster = n

        def calcular_centroide(cluster):
            centroides = []
            for j in range(len(self.matriz[0])):
                valores = []
                for idx in cluster:
                    try:
                        valor = float(self.matriz[idx][j])  # Convertir a número si es posible
                        valores.append(valor)
                    except ValueError:
                        continue  # Ignorar valores no numéricos
                if valores:
                    centroides.append(np.mean(valores))  # Calcular la media de los valores numéricos
                else:
                    centroides.append(0)  # Reemplazar con 0 si no hay valores numéricos
            return np.array(centroides)

        while len(clusters) > 1:
            min_dist = float('inf')
            clust1, clust2 = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    centroide_1 = calcular_centroide(clusters[i])
                    centroide_2 = calcular_centroide(clusters[j])
                    distancia = np.sqrt(np.sum((centroide_1 - centroide_2) ** 2))
                    if distancia < min_dist:
                        min_dist = distancia
                        clust1, clust2 = i, j

            linkage_matrix.append([cluster_ids[clusters[clust1][0]],
                                   cluster_ids[clusters[clust2][0]],
                                   min_dist,
                                   len(clusters[clust1]) + len(clusters[clust2])])

            nuevo_cluster = clusters[clust1] + clusters[clust2]
            for idx in nuevo_cluster:
                cluster_ids[idx] = current_cluster
            clusters = [clusters[k] for k in range(len(clusters)) if k != clust1 and k != clust2]
            clusters.append(nuevo_cluster)
            current_cluster += 1

        return np.array(linkage_matrix)

def main(matriz):
    try:
        # Crear instancia de la clase
        cluster = ClusterEslabonamiento(matriz)

        # Validar la matriz antes de procesarla
        matriz_validada = cluster.matriz  # Matriz ya validada en el constructor
        cluster.mostrar_matriz("Matriz validada", matriz_validada)

        if matriz_validada:
            matriz = cluster.eliminar_columnas_constantes(matriz_validada)
            cluster.mostrar_matriz("Matriz después de eliminar columnas constantes", matriz)

            media_datos = cluster.calcular_media(matriz)
            cluster.mostrar_matriz("Media de las columnas", [media_datos])  # Imprime como una fila

            matriz_centrada = cluster.centrar_matriz(matriz, media_datos)
            cluster.mostrar_matriz("Matriz centrada", matriz_centrada)

            matriz_covarianza = cluster.calcular_covarianza(matriz_centrada)
            cluster.mostrar_matriz("Matriz de covarianza", matriz_covarianza)

            matriz_cov_inversa = cluster.calcular_inversa_covarianza(matriz_covarianza)
            if matriz_cov_inversa is not None:
                cluster.mostrar_matriz("Inversa de la matriz de covarianza", matriz_cov_inversa)

                # Crear la matriz de distancias
                distancias = []
                for vector in matriz:
                    distancia = cluster.calcular_distancia_mahalanobis_manual(vector, media_datos, matriz_cov_inversa)
                    distancias.append([round(float(distancia), 3) for _ in range(len(matriz))])  # Rellenar la fila con la misma distancia

                cluster.mostrar_matriz("Matriz de distancias de Mahalanobis", distancias)

                cluster.elegir_eslabonamiento(distancias)


            else:
                print("No se pudieron calcular distancias debido a problemas con la matriz de covarianza.")
        else:
            print("No se pudo cargar la matriz correctamente.")
    except ValueError as e:
        print(f"Error en la matriz: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
