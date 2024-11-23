import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

class Agrupador:
    def __init__(self, matriz):
        self.matriz = matriz
        self.historial_eslabonamientos = []

    def calcular_distancia_gower(self):
        n = len(self.matriz)
        matriz_distancia = np.zeros((n, n))
        columnas = len(self.matriz[0])

        for i in range(n):
            for j in range(n):
                distancia = 0
                for k in range(columnas):
                    try:
                        # Intentamos tratar el dato como numérico
                        distancia += abs(float(self.matriz[i][k]) - float(self.matriz[j][k]))
                    except ValueError:
                        # Si no es numérico, tratamos como categórico
                        distancia += int(self.matriz[i][k] != self.matriz[j][k])
                matriz_distancia[i, j] = distancia / columnas

        return matriz_distancia

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
    agrupador = Agrupador(matriz)
    matriz_distancia = agrupador.calcular_distancia_gower()

    print("Matriz de distancia de Gower:")
    for fila in matriz_distancia:
        print(["{:.4f}".format(valor) for valor in fila])

    agrupador.elegir_eslabonamiento(matriz_distancia)
