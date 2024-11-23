import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

def leer_matriz(archivo):
    datos = []
    try:
        with open(archivo, 'r') as file:
            for linea in file:
                fila = linea.strip().split(',')
                datos.append([x.strip() for x in fila])
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None

    n_columnas = len(datos[0])
    for fila in datos:
        if len(fila) != n_columnas:
            print("Error: La matriz tiene filas con diferente número de columnas.")
            return None

    return datos

def identificar_tipo(datos):
    tipos = []
    for col in range(len(datos[0])):
        es_numerica = True
        for fila in datos:
            try:
                float(fila[col])
            except ValueError:
                es_numerica = False
                break
        tipos.append("numérica" if es_numerica else "categórica")
    return tipos

def normalizar_numéricas(datos, indices_num):
    min_valores = [float("inf")] * len(indices_num)
    max_valores = [-float("inf")] * len(indices_num)

    for fila in datos:
        for i, idx in enumerate(indices_num):
            valor = float(fila[idx])
            if valor < min_valores[i]:
                min_valores[i] = valor
            if valor > max_valores[i]:
                max_valores[i] = valor

    for fila in datos:
        for i, idx in enumerate(indices_num):
            valor = float(fila[idx])
            if max_valores[i] - min_valores[i] != 0:
                fila[idx] = (valor - min_valores[i]) / (max_valores[i] - min_valores[i])
            else:
                fila[idx] = 0.0

def calcular_distancia_gower(datos, tipos):
    n = len(datos)
    matriz_distancia = np.zeros((n, n))
    indices_num = [i for i, tipo in enumerate(tipos) if tipo == "numérica"]
    indices_cat = [i for i, tipo in enumerate(tipos) if tipo == "categórica"]

    for i in range(n):
        for j in range(n):
            distancia = 0
            for k in indices_num:
                distancia += abs(float(datos[i][k]) - float(datos[j][k]))
            for k in indices_cat:
                distancia += int(datos[i][k] != datos[j][k])
            matriz_distancia[i, j] = distancia / len(tipos)

    return matriz_distancia

def mostrar_matriz(matriz):
    np.set_printoptions(precision=4, suppress=True)
    for fila in matriz:
        print([f"{valor:.4f}" for valor in fila])

def decidir_agrupamiento(matriz_distancia):
    promedio_distancia = np.mean(matriz_distancia)
    if promedio_distancia < 0.5:
        metodo = "single"
        tipo_agrupamiento = "Similitud"
        criterio = "El promedio de distancias es bajo, por lo que se usará agrupamiento por similitud."
    else:
        metodo = "complete"
        tipo_agrupamiento = "Disimilitud"
        criterio = "El promedio de distancias es alto, por lo que se usará agrupamiento por disimilitud."

    return metodo, tipo_agrupamiento, criterio

def generar_historial_agrupamientos(linkage_matrix):
    historial = []
    n = len(linkage_matrix) + 1

    for idx1, idx2, dist, _ in linkage_matrix:
        elemento1 = int(idx1) if idx1 < n else f"({historial[int(idx1 - n)]})"
        elemento2 = int(idx2) if idx2 < n else f"({historial[int(idx2 - n)]})"
        nuevo_grupo = f"{elemento1}, {elemento2}"
        historial.append(nuevo_grupo)
        print(f"Se agrupa {elemento1} con {elemento2} a una distancia de {dist:.4f}")

def agrupar_y_mostrar_dendograma(matriz_distancia):
    matriz_distancia_condensada = squareform(matriz_distancia)
    metodo, tipo_agrupamiento, criterio = decidir_agrupamiento(matriz_distancia)

    print(f"Tipo de agrupamiento: {tipo_agrupamiento}")
    print(f"Criterio utilizado: {criterio}")

    linkage = sch.linkage(matriz_distancia_condensada, method=metodo)
    generar_historial_agrupamientos(linkage)

    plt.figure(figsize=(10, 7))
    sch.dendrogram(linkage)
    plt.title(f"Dendrograma - Agrupamiento por {tipo_agrupamiento}")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Distancia")
    plt.show()

def main():
    archivo = input("Ingresa el nombre del archivo .txt: ")
    datos = leer_matriz(archivo)

    if datos is None:
        return

    tipos = identificar_tipo(datos)
    print(f"Tipos de columnas: {tipos}")

    indices_num = [i for i, tipo in enumerate(tipos) if tipo == "numérica"]
    normalizar_numéricas(datos, indices_num)

    matriz_distancia = calcular_distancia_gower(datos, tipos)
    print("Matriz de distancia de Gower:")
    mostrar_matriz(matriz_distancia)

    agrupar_y_mostrar_dendograma(matriz_distancia)

if __name__ == "__main__":
    main()
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
