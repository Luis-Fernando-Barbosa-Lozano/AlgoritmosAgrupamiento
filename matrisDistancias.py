class EsquemaBinario:
    def __init__(self):
        self.matriz = []
        self.bandera_cargado = False  # Indicador para verificar si la matriz se ha cargado

    def cargar_matriz_desde_archivo(self, archivo):
        try:
            with open(archivo, 'r') as file:
                for linea in file:
                    valores = list(map(int, linea.split()))
                    # Verifica si se ha alcanzado el límite antes de añadir la fila
                    total_datos = sum(len(fila) for fila in self.matriz)  # Total de datos en la matriz actual
                    if total_datos + len(valores) >= 10000:
                        print("Error: El archivo contiene más de 10,000 registros. No se puede procesar.")
                        return  # Salir del método si se excede el límite

                    self.matriz.append(valores)

            self.bandera_cargado = True  # Se cargó exitosamente
        except FileNotFoundError:
            print(f"Error: El archivo {archivo} no se encontró.")
        except ValueError as e:
            print(f"Error: El archivo contiene valores no válidos. Detalle: {e}")
        except Exception as e:
            print(f"Error inesperado: {e}")  # Captura cualquier otro error inesperado

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

        n = len(self.matriz)
        print("\nMatriz de distancias:")
        matriz_distancias = [[0] * n for _ in range(n)]  # Inicializa la matriz de distancias
        for i in range(n):
            for j in range(i + 1, n):  # Solo calculamos para j > i
                if j < n - 1:  # Verifica que j esté dentro del rango de la matriz
                    if metodo == "sokal":
                        distancia = self.calcular_indice_sokal(self.matriz[i], self.matriz[j])
                    elif metodo == "jaccard":
                        distancia = self.calcular_indice_jaccard(self.matriz[i], self.matriz[j])
                    else:
                        print("Método no reconocido. Por favor, elige 'sokal' o 'jaccard'.")
                        return

                    matriz_distancias[j][i] = distancia
                    print(f"Distancia entre {i + 1} y {j + 1}: {distancia:.2f}")  # Cambiado a números
        # Llenar la diagonal con 1
        for i in range(len(matriz_distancias)):
            matriz_distancias[i][i] = 1

        # Imprimir la matriz de distancias completa
        print("\nMatriz de distancias completa:")
        for i in range(len(matriz_distancias) - 1):  # Asegúrate de no imprimir la última fila
            # Solo imprime si hay una relación, que sería para las filas relevantes
            if i < len(matriz_distancias):  # Asegura que i esté en el rango de la matriz
                # Imprime la fila, reemplazando los valores por encima de la diagonal con asteriscos
                fila_imprimir = []
                for j in range(len(matriz_distancias) - 1):  # Asegúrate de no imprimir la última columna
                    if j > i:  # Si está por encima de la diagonal
                        fila_imprimir.append('*')
                    else:
                        fila_imprimir.append(f"{matriz_distancias[i][j]:.2f}")

                print("\t".join(fila_imprimir))

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
