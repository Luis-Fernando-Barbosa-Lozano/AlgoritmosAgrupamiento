import csv


def leer_datos(archivo):
    datos = []
    try:
        with open(archivo, 'r') as f:
            lector = csv.reader(f)
            for fila in lector:
                datos.append([float(valor.strip()) for valor in fila])
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
    return datos


def calcular_coeficiente_gower(datos):
    n = len(datos)
    m = len(datos[0]) if n > 0 else 0
    matriz_gower = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            suma_distancias = 0.0
            suma_pesos = 0.0

            for k in range(m):
                xi = datos[i][k]
                xj = datos[j][k]

                if es_binaria(datos, k):
                    distancia = 0.0 if xi == xj else 1.0
                else:
                    rango = calcular_rango(datos, k)
                    distancia = 0.0 if rango == 0 else abs(xi - xj) / rango

                suma_distancias += distancia
                suma_pesos += 1.0  # Asume peso igual para todas las variables

            similitud = 1.0 - (suma_distancias / suma_pesos)
            matriz_gower[i][j] = similitud
            matriz_gower[j][i] = similitud
        matriz_gower[i][i] = 1.0

    return matriz_gower


def es_binaria(datos, columna):
    return len(set(fila[columna] for fila in datos)) == 2


def calcular_rango(datos, columna):
    valores = [fila[columna] for fila in datos]
    return max(valores) - min(valores)


def realizar_agrupamiento(matriz_similitud, metodo):
    n = len(matriz_similitud)
    distancias = convertir_similitud_a_distancia(matriz_similitud)

    clusters = [str(i + 1) for i in range(n)]

    for paso in range(1, n):
        menor_distancia = float('inf')
        cluster1, cluster2 = -1, -1

        for i in range(len(distancias)):
            for j in range(i + 1, len(distancias[i])):
                if distancias[i][j] < menor_distancia:
                    menor_distancia = distancias[i][j]
                    cluster1, cluster2 = i, j

        print(
            f"Paso {paso}: Fusión de ({clusters[cluster1]}, {clusters[cluster2]}) con distancia {menor_distancia:.4f} [{metodo}]")

        for i in range(len(distancias)):
            if i != cluster1 and i != cluster2:
                distancias[cluster1][i] = calcular_nueva_distancia(distancias[cluster1][i], distancias[cluster2][i],
                                                                   metodo)
                distancias[i][cluster1] = distancias[cluster1][i]

        distancias[cluster1][cluster1] = 0

        clusters[cluster1] = f"({clusters[cluster1]},{clusters[cluster2]})"
        clusters.pop(cluster2)

        distancias = eliminar_columna_y_fila(distancias, cluster2)

        print(f"==== Matriz de Distancias Intermedia ({metodo}) ====")
        imprimir_matriz(distancias, clusters)
        print("=========================")


def eliminar_columna_y_fila(matriz, indice):
    n = len(matriz)
    nueva_matriz = [[0.0] * (n - 1) for _ in range(n - 1)]

    for i in range(n):
        if i == indice:
            continue  # Omitir la fila eliminada
        for j in range(n):
            if j == indice:
                continue  # Omitir la columna eliminada
            # Ajustamos las posiciones para copiar los valores correctamente
            ni = i - 1 if i > indice else i
            nj = j - 1 if j > indice else j
            nueva_matriz[ni][nj] = matriz[i][j]

    return nueva_matriz


def convertir_similitud_a_distancia(matriz_similitud):
    n = len(matriz_similitud)
    matriz_distancia = [[1.0 - matriz_similitud[i][j] for j in range(n)] for i in range(n)]
    return matriz_distancia


def calcular_nueva_distancia(d1, d2, metodo):
    if metodo == "centroide":
        return (d1 + d2) / 2
    elif metodo == "maximos":
        return max(d1, d2)
    elif metodo == "minimos":
        return min(d1, d2)
    else:
        return (d1 + d2) / 2  # Default (centroide)


def imprimir_matriz(matriz, clusters):
    print("      ", end="")
    for cluster in clusters:
        print(f"{cluster:<8}", end="")
    print()

    for i, cluster in enumerate(clusters):
        print(f"{cluster:<6}", end="")
        for j in range(len(clusters)):
            if i == j:
                print("0.00    ", end="")
            else:
                print(f"{matriz[i][j]:.2f}    ", end="")
        print()


def main():
    archivo = input("Ingresa el archivo CSV con datos: ")
    datos = leer_datos(archivo)

    if not datos:
        print("El archivo está vacío o no tiene datos válidos.")
        return

    print("Calculando matriz de Gower...")
    matriz_gower = calcular_coeficiente_gower(datos)

    metodo = input("Selecciona el método de agrupamiento (centroide, maximos, minimos): ").strip().lower()
    if metodo not in ["centroide", "maximos", "minimos"]:
        print("Método no válido. Se usará 'centroide' por defecto.")
        metodo = "centroide"

    print("Realizando agrupamiento jerárquico...")
    realizar_agrupamiento(matriz_gower, metodo)

    print("Proceso completado.")


if __name__ == "__main__":
    main()