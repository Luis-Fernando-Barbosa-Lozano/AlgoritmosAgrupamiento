import math

class EsquemaBinario:
    def __init__(self):
        self.matriz = []
        self.bandera_cargado = False
        self.historial_eslabonamientos = []  # Lista para almacenar los eslabonamientos realizados

    def cargar_matriz_desde_archivo(self, archivo):
        try:
            with open(archivo, 'r') as file:
                for linea in file:
                    valores = list(map(int, linea.split()))
                    total_datos = sum(len(fila) for fila in self.matriz)
                    if total_datos + len(valores) >= 10000:
                        print("Error: El archivo contiene más de 10,000 registros. No se puede procesar.")
                        return
                    self.matriz.append(valores)

            self.bandera_cargado = True
        except FileNotFoundError:
            print(f"Error: El archivo {archivo} no se encontró.")
        except ValueError as e:
            print(f"Error: El archivo contiene valores no válidos. Detalle: {e}")
        except Exception as e:
            print(f"Error inesperado: {e}")

        # Verificar la matriz cargada
        print("Matriz cargada:", self.matriz)
        print("Número de filas:", len(self.matriz))

    def imprimir_matriz(self):
        if not self.bandera_cargado:
            print("La matriz no se ha cargado. No se puede imprimir.")
            return
        for fila in self.matriz:
            print("\t".join(map(str, fila)))

    def calcular_distancias(self, metodo):
        if not self.bandera_cargado:
            print("La matriz no se ha cargado. No se puede calcular distancias.")
            return

        n = len(self.matriz)  # El número de filas (individuos)
        print("\nMatriz de distancias:")
        matriz_distancias = [[0] * n for _ in range(n)]  # Inicializa la matriz de distancias de nxn

        for i in range(n):
            for j in range(i + 1, n):  # Solo calculamos para j > i
                if metodo == "sokal":
                    distancia = self.calcular_indice_sokal(self.matriz[i], self.matriz[j])
                elif metodo == "jaccard":
                    distancia = self.calcular_indice_jaccard(self.matriz[i], self.matriz[j])
                else:
                    print("Método no reconocido. Por favor, elige 'sokal' o 'jaccard'.")
                    return

                matriz_distancias[i][j] = distancia
                matriz_distancias[j][i] = distancia  # Simetría en la matriz de distancias
                print(f"Distancia entre {i + 1} y {j + 1}: {distancia:.2f}")

        # Llenar la diagonal con 1
        for i in range(n):
            matriz_distancias[i][i] = 1

        # Imprimir la matriz de distancias completa
        print("\nMatriz de distancias completa:")
        for i in range(n):
            fila_imprimir = []
            for j in range(n):
                if j > i:
                    fila_imprimir.append('*')
                else:
                    fila_imprimir.append(f"{matriz_distancias[i][j]:.2f}")

            print("\t".join(fila_imprimir))

        self.elegir_eslabonamiento(matriz_distancias)

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
        a_count = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
        b_count = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        c_count = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        d_count = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
        return a_count, b_count, c_count, d_count

    def calcular_indice_sokal(self, a, b):
        a_count, b_count, c_count, d_count = self.calcular_conteo(a, b)
        return round((a_count + d_count) / (a_count + b_count + c_count + d_count), 2) if (a_count + b_count + c_count + d_count) != 0 else 0.0

    def calcular_indice_jaccard(self, a, b):
        a_count, b_count, c_count, d_count = self.calcular_conteo(a, b)
        return round(a_count / (a_count + b_count + c_count), 2) if (a_count + b_count + c_count) != 0 else 0.0

    def elegir_metodo(self):
        while True:
            metodo = input("¿Deseas usar el índice de Sokal o Jaccard? (sokal/jaccard): ").strip().lower()
            if metodo in ["sokal", "jaccard"]:
                return metodo
            else:
                print("Opción no válida. Por favor elige 'sokal' o 'jaccard'.")

if __name__ == "__main__":
    esquema = EsquemaBinario()
    esquema.cargar_matriz_desde_archivo("matriz.txt")
    if esquema.bandera_cargado:
        esquema.imprimir_matriz()
        metodo_elegido = esquema.elegir_metodo()
        esquema.calcular_distancias(metodo_elegido)
