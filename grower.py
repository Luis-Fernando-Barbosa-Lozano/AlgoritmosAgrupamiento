import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def _imprimir_matriz(matriz, etiquetas, paso):
    print(f"\nMatriz parcial después del paso {paso}:")
    print("   " + "  ".join(etiquetas))
    for i, fila in enumerate(matriz):
        print(f"{etiquetas[i]}: " + "  ".join(f"{v:.4f}" for v in fila))


class Agrupador:
    def __init__(self, matriz):
        self.matriz = matriz
        self.historial_eslabonamientos = []
        self.columnas_min = []
        self.columnas_max = []
        self.normalizar_matriz()

    def normalizar_matriz(self):
        columnas = len(self.matriz[0])
        self.columnas_min = []
        self.columnas_max = []

        for j in range(columnas):
            try:
                columna = [float(fila[j]) for fila in self.matriz]
                col_min = min(columna)
                col_max = max(columna)
                self.columnas_min.append(col_min)
                self.columnas_max.append(col_max)

                for i in range(len(self.matriz)):
                    self.matriz[i][j] = (float(self.matriz[i][j]) - col_min) / (col_max - col_min)
            except ValueError:
                self.columnas_min.append(None)
                self.columnas_max.append(None)

    def calcular_distancia_gower(self):
        n = len(self.matriz)
        columnas = len(self.matriz[0])
        matriz_distancia = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                distancia = 0
                for k in range(columnas):
                    try:
                        distancia += abs(float(self.matriz[i][k]) - float(self.matriz[j][k]))
                    except ValueError:
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

        self.generar_dendrograma(matriz_distancias, metodo=metodo_dendrograma)

    def eslabonamiento_vecino_mas_cercano(self, matriz_distancias):
        self._eslabonamiento_generico(matriz_distancias, min, "Vecino más cercano")

    def eslabonamiento_vecino_mas_lejano(self, matriz_distancias):
        self._eslabonamiento_generico(matriz_distancias, max, "Vecino más lejano")

    def eslabonamiento_centroide(self, matriz_distancias):
        self._eslabonamiento_generico(matriz_distancias, np.mean, "Centroide")

    def _eslabonamiento_generico(self, matriz_distancias, func, tipo_eslabonamiento):
        paso = 1
        while len(clusters) > 1:
            print(f"\nClusters actuales (Paso {paso}): {clusters}")

            # Seleccionar los dos clusters más cercanos
            func = min if tipo_eslabonamiento == "Vecino más cercano" else max
            extrema_dist = float('inf') if func == min else -float('inf')
            clust1, clust2 = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distancias = [
                        matriz_distancias[a - 1][b - 1]
                        for a in clusters[i]
                        for b in clusters[j]
                    ]
                    distancia = func(distancias)
                    if func(distancia, extrema_dist) == distancia:
                        extrema_dist = distancia
                        clust1, clust2 = i, j

            # Verificar índices válidos
            if clust1 == -1 or clust2 == -1:
                print(f"Error: Índices clust1 ({clust1}), clust2 ({clust2}) no válidos.")
                break

            # Combinar los clusters seleccionados
            nuevo_cluster = clusters[clust1] + clusters[clust2]
            print(f"Combinar clusters {clusters[clust1]} y {clusters[clust2]} en {nuevo_cluster}")

            # Actualizar los clusters
            clusters = [clusters[i] for i in range(len(clusters)) if i not in [clust1, clust2]]
            clusters.append(nuevo_cluster)

            # Generar una nueva matriz de distancias
            nueva_matriz = []
            for i in range(len(clusters)):
                fila = []
                for j in range(len(clusters)):
                    if i == j:
                        fila.append(0)
                    else:
                        distancias = [
                            matriz_distancias[a - 1][b - 1]
                            for a in clusters[i]
                            for b in clusters[j]
                        ]
                        fila.append(func(distancias))
                nueva_matriz.append(fila)

            # Actualizar etiquetas y matriz de distancias
            etiquetas = [str(cluster) for cluster in clusters]
            matriz_distancias = nueva_matriz

            # Imprimir la matriz actualizada
            _imprimir_matriz(matriz_distancias, etiquetas, paso)
            paso += 1

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
                        valor = float(self.matriz[idx][j])
                        valores.append(valor)
                    except ValueError:
                        continue
                if valores:
                    centroides.append(np.mean(valores))
                else:
                    centroides.append(0)
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
    n = len(matriz_distancia)

    etiquetas = [str(i + 1) for i in range(n)]
    _imprimir_matriz(matriz_distancia, etiquetas, paso=0)

    agrupador.elegir_eslabonamiento(matriz_distancia)