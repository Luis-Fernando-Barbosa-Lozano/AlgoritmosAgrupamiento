import math
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

class EsquemaBinario:
    def __init__(self, matriz_distancias=None):
        if matriz_distancias is not None:
            self.matriz = [[int(valor) for valor in fila] for fila in matriz_distancias]
        else:
            self.matriz = []
        self.bandera_cargado = False
        self.historial_eslabonamientos = []  # Lista para almacenar los eslabonamientos realizados

    def imprimir_matriz(self):
        if not self.bandera_cargado:
            print("La matriz no se ha cargado. No se puede imprimir.")
            return
        for fila in self.matriz:
            print("\t".join(map(str, fila)))

    def generar_dendrograma(self, metodo):
        if not self.bandera_cargado:
            print("La matriz no se ha cargado. No se puede generar el dendrograma.")
            return

        # Calcula las distancias utilizando el método que prefieras
        matriz_distancias = self.calcular_distancias(metodo)  # Guarda la matriz de distancias

        if matriz_distancias is None:  # Comprueba si hubo un error en el cálculo de distancias
            return

        # Convierte la matriz de distancias a formato condensado
        matriz_distancias_condensada = squareform(matriz_distancias)

        # Usa la función linkage para realizar el agrupamiento
        Z = linkage(matriz_distancias_condensada, method='average')

        # Historial de eslabonamientos basado en el formato de linkage
        self.historial_eslabonamientos.clear()  # Limpiar historial previo
        for i, (clust1, clust2, dist, _) in enumerate(Z):
            self.historial_eslabonamientos.append(
                f"Paso {i + 1}: Combinar clúster {int(clust1 + 1)} y clúster {int(clust2 + 1)} con distancia {dist:.2f}"
            )

        # Dibuja el dendrograma
        plt.figure(figsize=(10, 7))
        dendrogram(Z, labels=[f'Individuo {i + 1}' for i in range(len(self.matriz))])
        plt.title('Dendrograma')
        plt.xlabel('Individuos')
        plt.ylabel('Distancia')
        plt.show()

    def calcular_distancias(self, metodo):
        n = len(self.matriz)
        matriz_distancias = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if metodo == "sokal":
                    distancia = self.calcular_indice_sokal(self.matriz[i], self.matriz[j])
                elif metodo == "jaccard":
                    distancia = self.calcular_indice_jaccard(self.matriz[i], self.matriz[j])
                else:
                    print("Método no reconocido. Por favor, elige 'sokal' o 'jaccard'.")
                    return None

                matriz_distancias[i][j] = distancia
                matriz_distancias[j][i] = distancia

        print("\nMatriz de distancias completa:")
        for fila in matriz_distancias:
            print("\t".join(f"{valor:.2f}" for valor in fila))

        return matriz_distancias

    def elegir_eslabonamiento(self, matriz_distancias):
        print("\nElige el tipo de eslabonamiento:")
        print("1. Vecino más cercano")
        print("2. Vecino más lejano")
        print("3. Centroide")

        tipo = input("Introduce el número correspondiente: ").strip()

        if tipo == "1":
            self.eslabonamiento_vecino_mas_cercano(matriz_distancias)
        elif tipo == "2":
            self.eslabonamiento_vecino_mas_lejano(matriz_distancias)
        elif tipo == "3":
            self.eslabonamiento_centroide(matriz_distancias)
        else:
            print("Opción no válida. Intenta de nuevo.")
            self.elegir_eslabonamiento(matriz_distancias)

    def eslabonamiento_vecino_mas_cercano(self, matriz_distancias):
        n = len(matriz_distancias)
        clusters = [[i + 1] for i in range(n)]  # Cada punto comienza como un clúster separado

        while len(clusters) > 1:
            # Encontrar la distancia mínima entre clústeres
            min_dist = float('inf')
            clust1, clust2 = -1, -1
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    for a in clusters[i]:
                        for b in clusters[j]:
                            if matriz_distancias[a - 1][b - 1] < min_dist:  # Restar 1 para índices de matriz
                                min_dist = matriz_distancias[a - 1][b - 1]
                                clust1, clust2 = i, j

            # Combinar los clústeres encontrados
            nuevo_cluster = clusters[clust1] + clusters[clust2]
            self.historial_eslabonamientos.append(
                f"Combinar elemento {clusters[clust1]} y elemento {clusters[clust2]} con distancia {min_dist:.2f}\n"
            )

            # Actualizar la lista de clústeres
            clusters = [clusters[k] for k in range(len(clusters)) if k != clust1 and k != clust2]
            clusters.append(nuevo_cluster)

            # Recalcular distancias para el nuevo clúster (vecino más cercano)
            for i in range(len(matriz_distancias)):
                if i != clust1 and i != clust2:
                    matriz_distancias[i][len(clusters) - 1] = min(matriz_distancias[i][clust1],
                                                                  matriz_distancias[i][clust2])

        # Al imprimir, cada eslabonamiento se mostrará en una nueva línea
        print("\nHistorial de eslabonamientos:\n")
        for eslabonamiento in self.historial_eslabonamientos:
            print(eslabonamiento)

    def eslabonamiento_vecino_mas_lejano(self, matriz_distancias):
        n = len(matriz_distancias)
        clusters = [[i + 1] for i in range(n)]  # Cada punto comienza como un clúster separado

        while len(clusters) > 1:
            # Encontrar la distancia máxima entre clústeres
            max_dist = -float('inf')
            clust1, clust2 = -1, -1
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    for a in clusters[i]:
                        for b in clusters[j]:
                            if matriz_distancias[a - 1][b - 1] > max_dist:  # Restar 1 para índices de matriz
                                max_dist = matriz_distancias[a - 1][b - 1]
                                clust1, clust2 = i, j

            # Combinar los clústeres encontrados
            nuevo_cluster = clusters[clust1] + clusters[clust2]
            self.historial_eslabonamientos.append(
                f"Combinar elemento {clusters[clust1]} y elemento {clusters[clust2]} con distancia {max_dist:.2f}\n"
            )

            # Actualizar la lista de clústeres
            clusters = [clusters[k] for k in range(len(clusters)) if k != clust1 and k != clust2]
            clusters.append(nuevo_cluster)

            # Recalcular distancias para el nuevo clúster (vecino más lejano)
            for i in range(len(matriz_distancias)):
                if i != clust1 and i != clust2:
                    matriz_distancias[i][len(clusters) - 1] = max(matriz_distancias[i][clust1],
                                                                  matriz_distancias[i][clust2])

        # Al imprimir, cada eslabonamiento se mostrará en una nueva línea
        print("\nHistorial de eslabonamientos:\n")
        for eslabonamiento in self.historial_eslabonamientos:
            print(eslabonamiento)

    def eslabonamiento_centroide(self, matriz_distancias):
        n = len(matriz_distancias)
        clusters = [[i + 1] for i in range(n)]  # Cada punto comienza como un clúster separado

        def calcular_centroide(cluster):
            """ Calcula el centroide promedio del clúster """
            centroide = [sum(self.matriz[idx - 1][j] for idx in cluster) / len(cluster) for j in
                         range(len(self.matriz[0]))]
            return centroide

        while len(clusters) > 1:
            min_dist = float('inf')
            clust1, clust2 = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    centroide_1 = calcular_centroide(clusters[i])
                    centroide_2 = calcular_centroide(clusters[j])

                    distancia = math.sqrt(sum((x - y) ** 2 for x, y in zip(centroide_1, centroide_2)))
                    if distancia < min_dist:
                        min_dist = distancia
                        clust1, clust2 = i, j

            # Combinar los clústeres encontrados
            nuevo_cluster = clusters[clust1] + clusters[clust2]
            self.historial_eslabonamientos.append(
                f"Combinar elemento {clusters[clust1]} y elemento {clusters[clust2]} con distancia {min_dist:.2f}\n"
            )

            # Actualizar la lista de clústeres
            clusters = [clusters[k] for k in range(len(clusters)) if k != clust1 and k != clust2]
            clusters.append(nuevo_cluster)

        # Al imprimir, cada eslabonamiento se mostrará en una nueva línea
        print("\nHistorial de eslabonamientos:\n")
        for eslabonamiento in self.historial_eslabonamientos:
            print(eslabonamiento)

    def calcular_conteo(self, a, b):
        print(f"Comparando: {a} y {b}")  # Imprime las filas comparadas
        a_count = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
        b_count = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        c_count = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        d_count = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
        print(f"Resultado de conteo - a_count: {a_count}, b_count: {b_count}, c_count: {c_count}, d_count: {d_count}")
        return a_count, b_count, c_count, d_count

    def calcular_indice_sokal(self, a, b):
        a_count, b_count, c_count, d_count = self.calcular_conteo(a, b)
        return round((a_count + d_count) / (a_count + b_count + c_count + d_count), 2) if (a_count + b_count + c_count + d_count) != 0 else 0.0

    def calcular_indice_jaccard(self, a, b):
        a_count, b_count, c_count, d_count = self.calcular_conteo(a, b)
        denominador = a_count + b_count + c_count
        print(f"Jaccard - Numerador: {a_count}, Denominador: {denominador}")
        return round(a_count / denominador, 2) if denominador != 0 else 0.0

    def elegir_metodo(self):
        while True:
            metodo = input("¿Deseas usar el índice de Sokal o Jaccard? (sokal/jaccard): ").strip().lower()
            if metodo in ["sokal", "jaccard"]:
                return metodo
            else:
                print("Opción no válida. Por favor elige 'sokal' o 'jaccard'.")

def main(matriz):
    # Crear una instancia de la clase EsquemaBinario
    esquema_binario = EsquemaBinario(matriz)

    # Activar la bandera para indicar que la matriz está cargada
    esquema_binario.bandera_cargado = True

    # Imprimir la matriz cargada
    print("Matriz inicial:")
    esquema_binario.imprimir_matriz()

    # Solicitar al usuario el método de cálculo de distancia
    metodo = esquema_binario.elegir_metodo()

    # Calcular la matriz de distancias con el método elegido
    matriz_distancia = esquema_binario.calcular_distancias(metodo)
    if matriz_distancia is None:  # Si hay error en el cálculo, finalizar el programa
        return

    # Preguntar al usuario por el tipo de eslabonamiento
    esquema_binario.elegir_eslabonamiento(matriz_distancia)

    # Generar el dendrograma
    esquema_binario.generar_dendrograma(metodo)

    # Mostrar historial de eslabonamientos
    print("\nHistorial de eslabonamientos:")
    for historia in esquema_binario.historial_eslabonamientos:
        print(historia)

    print("\nProceso completado.")
