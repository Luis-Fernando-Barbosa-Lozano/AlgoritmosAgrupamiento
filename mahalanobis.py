import numpy as np
import csv

def leer_archivo(file_path):
    datos = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            datos.append([float(valor) for valor in row])
    return datos


def calcular_matriz_covarianza(datos):
    datos = np.array(datos)
    medias = np.mean(datos, axis=0)
    covarianza = np.cov(datos, rowvar=False)
    return covarianza


def inversa_matriz(matriz):
    return np.linalg.inv(matriz)


def calcular_distancia_mahalanobis(x, y, inversa_covarianza):
    diff = np.array(x) - np.array(y)
    return np.sqrt(np.dot(np.dot(diff, inversa_covarianza), diff.T))


def calcular_matriz_mahalanobis(datos):
    n = len(datos)
    m = len(datos[0])

    matriz_covarianza = calcular_matriz_covarianza(datos)
    matriz_inversa_covarianza = inversa_matriz(matriz_covarianza)

    distancias = []
    for i in range(n):
        fila = []
        for j in range(n):
            if i == j:
                fila.append(0.0)
            else:
                distancia = calcular_distancia_mahalanobis(datos[i], datos[j], matriz_inversa_covarianza)
                fila.append(distancia)
        distancias.append(fila)

    return distancias


def guardar_historial_etiquetado_csv(matriz, etiquetas, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(etiquetas)
        for i in range(len(matriz)):
            if etiquetas[i]:
                writer.writerow([etiquetas[i]] + matriz[i])


def imprimir_matriz(mensaje, matriz):
    print(mensaje)
    for fila in matriz:
        print(" ".join([f"{valor:10.2f}" for valor in fila]))
    print("\n")


def imprimir_matriz_etiquetada(titulo, matriz, etiquetas):
    print(f"==== {titulo} ====")

    num_filas = len(matriz)
    if num_filas == 0:
        print("La matriz está vacía.")
        return

    print("      ", end="")
    for etiqueta in etiquetas:
        print(f"{etiqueta:<8}", end="")
    print()

    for i in range(num_filas):
        print(f"{etiquetas[i]:<6}", end="")
        for j in range(len(matriz[i])):
            print(f"{matriz[i][j]:<8.2f}", end="")
        print()
    print("=========================")


def ejecutar_eslabonamientos_generico(datos, metodo, output_file):
    distancias = [row[:] for row in datos]
    etiquetas = [str(i + 1) for i in range(len(distancias))]

    print(f"Matriz inicial de distancias ({metodo}):")
    imprimir_matriz_etiquetada(f"Distancias Iniciales ({metodo})", distancias, etiquetas)

    etapa = 1
    while len(etiquetas) > 1:
        mejor_valor = float('inf') if metodo == "minimos" else -float('inf')
        E1, E2 = -1, -1

        for i in range(len(distancias)):
            for j in range(i + 1, len(distancias[i])):
                valor_actual = distancias[i][j]
                if (metodo == "minimos" and valor_actual < mejor_valor) or \
                        (metodo == "maximos" and valor_actual > mejor_valor) or \
                        (metodo == "centroide" and valor_actual > mejor_valor):
                    mejor_valor = valor_actual
                    E1, E2 = i, j

        if E1 != -1 and E2 != -1:
            nueva_etiqueta = f"({etiquetas[E1]},{etiquetas[E2]})"
            etiquetas[E1] = nueva_etiqueta
            etiquetas.pop(E2)

            for i in range(len(distancias)):
                if i != E1 and i != E2:
                    nueva_distancia = 0.0
                    if metodo == "minimos":
                        nueva_distancia = min(distancias[E1][i], distancias[E2][i])
                    elif metodo == "maximos":
                        nueva_distancia = max(distancias[E1][i], distancias[E2][i])
                    elif metodo == "centroide":
                        nueva_distancia = (distancias[E1][i] + distancias[E2][i]) / 2
                    distancias[E1][i] = nueva_distancia
                    distancias[i][E1] = nueva_distancia

            distancias.pop(E2)
            for row in distancias:
                row.pop(E2)

            print(f"Paso {etapa}: Fusión de ({E1 + 1}, {E2 + 1}) con distancia {mejor_valor:.4f} [{metodo}]")
            imprimir_matriz_etiquetada(f"Matriz de Distancias Intermedia ({metodo})", distancias, etiquetas)
            etapa += 1

    guardar_historial_etiquetado_csv(distancias, etiquetas, output_file)
    print(f"Historial guardado en: {output_file}")


def main():
    input_file = input("Ingresa el archivo CSV con datos: ")
    datos = leer_archivo(input_file)

    if not datos:
        print("El archivo está vacío o no tiene datos válidos.")
        return

    print("Calculando matriz de Mahalanobis...")
    matriz_mahalanobis = calcular_matriz_mahalanobis(datos)
    imprimir_matriz("Matriz de Mahalanobis", matriz_mahalanobis)

    print("Calculando eslabonamientos...")
    ejecutar_eslabonamientos_generico(matriz_mahalanobis, "minimos", "historial_minimos_maha.csv")
    ejecutar_eslabonamientos_generico(matriz_mahalanobis, "maximos", "historial_maximos_maha.csv")
    ejecutar_eslabonamientos_generico(matriz_mahalanobis, "centroide", "historial_centroide_maha.csv")

    print("Ejecución completada.")


if __name__ == "__main__":
    main()