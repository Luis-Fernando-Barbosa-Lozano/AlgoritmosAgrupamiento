import matrisDistancias as md
from grower import main as main_grower
#from mahalanobis import main as main_mahalanobis

class ProcesadorMatriz:
    def __init__(self):
        self.archivo = None
        self.matriz = []
        self.tipo_columnas = []
        self.bandera_cargado = False

    def cargar_matriz_desde_archivo(self, archivo):
        try:
            if archivo.endswith('.txt') or archivo.endswith('.csv'):
                with open(archivo, 'r', encoding='utf-8-sig') as file:
                    filas = [linea.strip().split(',' if archivo.endswith('.csv') else None) for linea in file]

                # Validar estructura del archivo y caracteres especiales
                if not self.validar_estructura(filas):
                    return

                # Determinar el tipo de datos en base a todos los valores de cada columna
                self.tipo_columnas = self.determinar_tipo_columnas(filas)
                print("Tipos de columnas detectados:", self.tipo_columnas)

                # Validar cada fila según el tipo detectado
                for i, fila in enumerate(filas, start=1):
                    if not self.validar_fila(fila, i):
                        return

                self.matriz = filas
                print("Matriz cargada correctamente.")
                self.verificar_tipos_columnas()
                self.bandera_cargado = True
            else:
                print(f"Error: Tipo de archivo no soportado.")
                return

        except FileNotFoundError:
            print(f"Error: El archivo {archivo} no se encontró.")
        except ValueError as e:
            print(f"Error: El archivo contiene valores no válidos. Detalle: {e}")
        except Exception as e:
            print(f"Error inesperado: {e}")

    def validar_estructura(self, filas):
        """
        Verifica que todas las filas tengan el mismo número de columnas
        y que no haya caracteres especiales no permitidos.
        """
        num_columnas = len(filas[0])
        for i, fila in enumerate(filas, start=1):
            if len(fila) != num_columnas:
                print(f"Error: La línea {i} tiene un número incorrecto de columnas.")
                return False
            for valor in fila:
                if not self.validar_caracteres(valor):
                    print(f"Error: El valor '{valor}' en la línea {i} contiene caracteres no permitidos.")
                    return False
        return True

    def validar_caracteres(self, valor):
        """
        Verifica que un valor sea válido (alfanumérico, numérico, o símbolo permitido como '.' o '-').
        """
        if valor.isalnum():
            return True
        if valor.replace('.', '', 1).replace('-', '', 1).isdigit():  # Permitir números con '.' o '-'
            return True
        return False

    def determinar_tipo_columnas(self, matriz):
        """
        Determina el tipo de datos para cada columna en base a todos los valores en esa columna.
        Puede ser 'binario', 'numérico' o 'categórico'.
        """
        tipos = []
        for columna in zip(*matriz):  # Transponer para iterar sobre columnas
            if all(valor in ['0', '1'] for valor in columna):  # Todos los valores deben ser 0 o 1
                tipos.append('binario')
            elif all(self.es_numerico(valor) for valor in columna):  # Todos deben ser números válidos
                tipos.append('numérico')
            else:
                tipos.append('categórico')  # Si no es binario ni numérico, es categórico
        return tipos

    def es_numerico(self, valor):
        """
        Verifica si un valor es un número válido (entero o decimal).
        """
        try:
            float(valor)
            return True
        except ValueError:
            return False

    def validar_fila(self, fila, numero_linea):
        """
        Valida que cada valor de una fila coincida con el tipo esperado.
        """
        for i, (valor, tipo) in enumerate(zip(fila, self.tipo_columnas)):
            if tipo == 'binario' and valor not in ['0', '1']:
                print(f"Error: El valor '{valor}' en la línea {numero_linea}, columna {i + 1} no es binario.")
                return False
            elif tipo == 'numérico':
                try:
                    float(valor)
                except ValueError:
                    print(f"Error: El valor '{valor}' en la línea {numero_linea}, columna {i + 1} no es numérico.")
                    return False
            elif tipo == 'categórico' and not valor.isalpha():
                print(f"Error: El valor '{valor}' en la línea {numero_linea}, columna {i + 1} no es categórico.")
                return False
        return True

    def verificar_tipos_columnas(self):
        """
        Verifica que todas las columnas tengan el mismo tipo que la primera fila.
        Llama al módulo correspondiente según el tipo de datos.
        """
        tipos_unicos = set(self.tipo_columnas)

        if len(tipos_unicos) == 1:
            tipo_dominante = tipos_unicos.pop()
            if tipo_dominante == 'binario':
                print("El archivo es 100% binario. Llamando a Esquema Binario...")
                md.main(self.matriz)  # Aquí pasas la matriz ya cargada
            elif tipo_dominante == 'numérico':
                print("El archivo es 100% numérico. Llamando a mahalanobis...")
                main_mahalanobis(self.matriz)  # Aquí también pasas la matriz
        else:
            print("El archivo contiene columnas mixtas. Llamando a gower...")
            main_grower(self.matriz)  # Llamas a Gower pasándole la matriz ya procesada


if __name__ == "__main__":
    # Crear una instancia de la clase ProcesadorMatriz
    procesador = ProcesadorMatriz()

    # Solicitar al usuario que ingrese la ruta del archivo
    archivo = input("Por favor, ingrese la ruta del archivo que desea procesar: ")

    # Intentar cargar el archivo y determinar el tipo de datos
    procesador.cargar_matriz_desde_archivo(archivo)
