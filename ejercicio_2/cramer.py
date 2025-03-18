## cramer.py ##
# Aplica la regla de cramer para resolver sistemas de ecuaciones
# en una matriz que determinamos como mat, y un vector vec.

# importamos numpy y el módulo de álgebra lineal
import numpy as np
from numpy import linalg 

# definimos el tamaño de la matriz n*n
n = 3

def RempCol(Mat, Vec, j):
    """
    Toma una matriz y remplaza la j-ésima columna por
    el vector determinado Vec.

    Parámetros:
    -----------
    Mat : numpy.ndarray
        Una matriz bidimensional de n filas por n columnas
    Vec : numpy.ndarray
        Un arreglo unidimensional de largo n por el que va
        a ser reemplazada la columna elegida
    i   : int
        Un natural que determina la j-ésima columna a reemplazar

    Retorna:
    --------
    numpy.ndarray:
        Una matriz de m por n cuya j-ésima columna ha sido reemplazada
        por Vec.
    """

    # Inicializamos la matriz resultante
    MatR = np.zeros((n,n))

    # Si la matriz, vector y número de entrada
    # no cumplen con los parámetros documentados se levanta una excepción.
    assert(np.shape(Mat) == (n, n))
    assert(np.shape(Vec) == (n,))
    assert(j <= n)
    assert(j > 0)

    #Copiamos por columnas
    for p in range(n):
        #Si la columna es la j-ésima, la reemplazamos por Vec.
        if p == j-1:
            for q in range(n):
                MatR[q,p] = Vec[q]
        else:
            for q in range(n):
                MatR[q,p] = Mat[q,p]

        
    #Regresamos el resultante
    return MatR

def Cramer(Mat, Vec, j):
    """
    Aplica la regla de cramer para el sistema Mat*x = Vec
    devolviendo el j-ésimo elemento del vector x.

    Parámetros:
    -----------

    Mat : numpy.ndarray
        Una matriz bidimensional cuadrada de lado n
    Vec : numpy.ndarray
        Un vector unidimensional de largo n
    j   : int
        Un natural que determina el componente
        j-ésimo del vector resultante

    Retorna:
    --------

    float64
        Un número que es componente j-ésimo del vector x.
    """

    # Obtenemos el determinante de la matriz de entrada
    detA = np.linalg.det(A)

    # Si el determinante es cero no podemos continuar
    if detA == 0:
        raise Exception("Determinante igual a cero. No se puede computar con Cramer.")
    else:
        xj = np.linalg.det(RempCol(Mat, Vec, j)) / detA

    # Si se puede calcular lo regresa, si no lanza una excepción:
    return(xj)


# Ejemplo con la matriz A y el vector v
A = np.array([[2.0, 0.0, 4.0],
              [0.0, 3.0, 3.0],
              [1.0, 0.0, 0.0]])
v = np.array([3.0, 2.0, 1.0])

print("Matriz de entrada: \n", A)
print("Vector de entrada: \n", v)

print("Vamos a resolver el sistema conformado por estos dos:")

# Inicializamos el vector respuesta
Res = np.zeros(n)

# Computamos Cramer para cada entrada
for k in range(n):
    Res[k] = Cramer(A, v, k+1)

print("############\nResultado:\n", Res)
