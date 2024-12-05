import csv


def leer_datos(archivo):
    datos = []
    try:
        with open(archivo, 'r', encoding='utf-8-sig') as f:
            lector = csv.reader(f)
            for fila in lector:
                datos.append([int(valor.strip()) for valor in fila])  # Usando valores binarios
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
    return datos


def calcular_distancia_sokal(datos, i, j):
    a = b = c = d = 0
    for k in range(len(datos[i])):
        if datos[i][k] == 1 and datos[j][k] == 1:
            a += 1
        elif datos[i][k] == 1 and datos[j][k] == 0:
            b += 1
        elif datos[i][k] == 0 and datos[j][k] == 1:
            c += 1
        elif datos[i][k] == 0 and datos[j][k] == 0:
            d += 1

    denominador = a + b + c + d
    if denominador == 0:
        return 0  # Retorna distancia 0 si ambos son iguales en todos los valores 0
    return (b + c) / denominador


def calcular_distancia_jaccard(datos, i, j):
    a = b = c = 0
    for k in range(len(datos[i])):
        if datos[i][k] == 1 and datos[j][k] == 1:
            a += 1
        elif datos[i][k] == 1 and datos[j][k] == 0:
            b += 1
        elif datos[i][k] == 0 and datos[j][k] == 1:
            c += 1

    denominador = a + b + c
    if denominador == 0:
        return 0  # Retorna distancia 0 si ambos son iguales en todos los valores 0
    return (b + c) / denominador

def calcular_distancia_binaria(datos):
    n = len(datos)
    matriz_distancia = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # Calcula la distancia binaria: 0 si son iguales, 1 si son diferentes
            distancia = sum(1 for k in range(len(datos[0])) if datos[i][k] != datos[j][k])
            matriz_distancia[i][j] = distancia
            matriz_distancia[j][i] = distancia

    return matriz_distancia

def realizar_agrupamiento(matriz_distancia, metodo, tipo_distancia):
    n = len(matriz_distancia)
    clusters = [str(i + 1) for i in range(n)]  # Inicializa nombres de clusters

    for paso in range(1, n):
        menor_distancia = float('inf')
        cluster1, cluster2 = -1, -1

        # Encuentra los clusters más cercanos
        for i in range(len(matriz_distancia)):
            for j in range(i + 1, len(matriz_distancia[i])):
                distancia = matriz_distancia[i][j]
                if distancia < menor_distancia:
                    menor_distancia = distancia
                    cluster1, cluster2 = i, j

        # Imprime la fusión actual
        print(
            f"Paso {paso}: Fusión de ({clusters[cluster1]}, {clusters[cluster2]}) con distancia {menor_distancia:.4f} [{metodo}]")

        # Combina los clusters en la lista
        clusters[cluster1] = f"({clusters[cluster1]},{clusters[cluster2]})"
        clusters.pop(cluster2)  # Elimina el cluster fusionado

        # Actualiza las distancias
        for i in range(len(matriz_distancia)):
            if i != cluster1 and i != cluster2:
                matriz_distancia[cluster1][i] = calcular_nueva_distancia(
                    matriz_distancia[cluster1][i], matriz_distancia[cluster2][i], metodo
                )
                matriz_distancia[i][cluster1] = matriz_distancia[cluster1][i]

        # Elimina fila y columna correspondientes al cluster fusionado
        matriz_distancia = eliminar_columna_y_fila(matriz_distancia, cluster2)

        # Imprime la matriz de distancias actualizada
        print(f"==== Matriz de Distancias Intermedia ({metodo}) ====")
        imprimir_matriz(matriz_distancia, clusters)
        print("=========================")

def eliminar_columna_y_fila(matriz, indice):
    nueva_matriz = [
        [matriz[i][j] for j in range(len(matriz)) if j != indice]
        for i in range(len(matriz)) if i != indice
    ]
    return nueva_matriz

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
    archivo = input("Ingresa el archivo CSV con datos binarios: ")
    datos = leer_datos(archivo)

    if not datos:
        print("El archivo está vacío o no tiene datos válidos.")
        return

    print("Calculando matriz de distancias binarias...")
    matriz_distancia = calcular_distancia_binaria(datos)

    tipo_distancia = input("Selecciona el tipo de distancia (sokal, jaccard): ").strip().lower()
    if tipo_distancia not in ["sokal", "jaccard"]:
        print("Tipo de distancia no válido. Se usará 'binaria' por defecto.")
        tipo_distancia = "binaria"

    metodo = input("Selecciona el método de agrupamiento (centroide, maximos, minimos): ").strip().lower()
    if metodo not in ["centroide", "maximos", "minimos"]:
        print("Método no válido. Se usará 'centroide' por defecto.")
        metodo = "centroide"



    print("Realizando agrupamiento jerárquico...")
    realizar_agrupamiento(matriz_distancia, metodo, tipo_distancia)

    print("Proceso completado.")



if __name__ == "__main__":
    main()
