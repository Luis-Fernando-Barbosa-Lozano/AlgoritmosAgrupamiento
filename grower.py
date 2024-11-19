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
