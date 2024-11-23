import numpy as np

def leer_matriz_archivo(nombre_archivo):
    matriz = []
    try:
        with open(nombre_archivo, 'r') as archivo:
            for linea in archivo:
                # Verificar si la línea contiene valores distintos de 0 o 1
                if any(valor not in ['0', '1'] for valor in linea.split()):
                    # Si se encuentra un valor diferente de 0 o 1, se procede con la lectura normal
                    try:
                        fila = [float(valor) for valor in linea.split() if valor.strip()]
                        if len(fila) != len(linea.split()):
                            print(f"Error: Espacio en blanco o valor vacío encontrado en la línea -> '{linea.strip()}'")
                            return None
                        matriz.append(fila)
                    except ValueError:
                        print(f"Error: Línea inválida encontrada y omitida -> '{linea.strip()}'")
                        return None
                else:
                    # Si encontramos solo 1 y 0, no procesamos el archivo y mostramos el mensaje
                    print("Error: El archivo contiene valores binarios (0 y 1) y no se puede procesar.")
                    return None
    except FileNotFoundError:
        print(f"Error: El archivo {nombre_archivo} no fue encontrado.")
        return None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None
    return matriz

def mostrar_matriz(matriz):
    print("Matriz cargada:")
    for fila in matriz:
        print(fila)

def validar_matriz(matriz):
    if matriz is None:
        print("Error: No se pudo cargar la matriz correctamente.")
        return False

    longitud_filas = len(matriz[0])
    for fila in matriz:
        if len(fila) != longitud_filas:
            print(f"Error: Las filas de la matriz tienen diferentes longitudes.")
            return False
        for valor in fila:
            if not isinstance(valor, (int, float)):
                print(f"Error: Valor inválido encontrado en la matriz -> '{valor}'")
                return False
            if valor == "" or valor is None:
                print(f"Error: Espacio vacío encontrado en la matriz.")
                return False
    return True

def calcular_media(matriz):
    filas = len(matriz)
    columnas = len(matriz[0])
    media = [0] * columnas

    for col in range(columnas):
        suma = sum(fila[col] for fila in matriz)
        media[col] = suma / filas
    return media

def centrar_matriz(matriz, media):
    matriz_centrada = []
    for fila in matriz:
        matriz_centrada.append([fila[i] - media[i] for i in range(len(fila))])
    return matriz_centrada

def eliminar_columnas_constantes(matriz):
    columnas_a_eliminar = []
    columnas = len(matriz[0])

    for col in range(columnas):
        valores_columna = [fila[col] for fila in matriz]
        if max(valores_columna) == min(valores_columna):
            columnas_a_eliminar.append(col)

    matriz_filtrada = []
    for fila in matriz:
        nueva_fila = [valor for i, valor in enumerate(fila) if i not in columnas_a_eliminar]
        matriz_filtrada.append(nueva_fila)

    return matriz_filtrada

def calcular_covarianza(matriz_centrada):
    filas = len(matriz_centrada)
    columnas = len(matriz_centrada[0])
    covarianza = [[0] * columnas for _ in range(columnas)]

    for i in range(columnas):
        for j in range(columnas):
            suma = sum(matriz_centrada[k][i] * matriz_centrada[k][j] for k in range(filas))
            covarianza[i][j] = suma / (filas - 1)
    return covarianza

def calcular_inversa_covarianza(matriz_cov):
    try:
        matriz_cov_inv = np.linalg.inv(matriz_cov)
        return matriz_cov_inv
    except np.linalg.LinAlgError:
        print("Error: La matriz de covarianza es singular y no se puede invertir.")
        return None

def calcular_distancia_mahalanobis_manual(vector, media, cov_inversa):
    resta = [vector[i] - media[i] for i in range(len(vector))]
    resta = np.array(resta)
    distancia = np.sqrt(resta.T @ cov_inversa @ resta)
    return distancia

def main(nombre_archivo):
    # Leer y mostrar matriz del archivo
    matriz_datos = leer_matriz_archivo(nombre_archivo)
    if matriz_datos:
        mostrar_matriz(matriz_datos)

        # Validar la matriz antes de continuar
        if not validar_matriz(matriz_datos):
            print("Error: La matriz contiene valores inválidos y no se puede procesar.")
        else:
            # Eliminar columnas constantes
            matriz_datos = eliminar_columnas_constantes(matriz_datos)

            # Calcular media y matriz centrada
            media_datos = calcular_media(matriz_datos)
            matriz_centrada = centrar_matriz(matriz_datos, media_datos)

            # Calcular matriz de covarianza e inversa
            matriz_covarianza = calcular_covarianza(matriz_centrada)
            matriz_cov_inversa = calcular_inversa_covarianza(matriz_covarianza)

            # Verificar si se pudo calcular la inversa
            if matriz_cov_inversa is not None:
                distancias = []
                for vector in matriz_datos:
                    distancia = float(calcular_distancia_mahalanobis_manual(vector, media_datos, matriz_cov_inversa))
                    distancias.append(distancia)

                # Mostrar resultados
                print("Media:", media_datos)
                print("Matriz de covarianza:", matriz_covarianza)
                print("Inversa de la matriz de covarianza:\n", matriz_cov_inversa)
                print("Distancias de Mahalanobis:", distancias)
            else:
                print("No se pudieron calcular distancias debido a problemas con la matriz de covarianza.")
    else:
        print("No se pudo cargar la matriz correctamente.")

