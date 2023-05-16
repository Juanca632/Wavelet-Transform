import numpy as np

def check_orthonormality(a,b):
    dot_product = np.dot(a, b)
    orthogonal = np.isclose(dot_product, 0)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    normalized = np.isclose(na, 1) and np.isclose(nb, 1)

    if orthogonal and normalized:
        print("The vectors are orthonormal.")
    else:
        print("The vectors are not orthonormal.")

# Signal
s = np.array([32, 32, 16, 8, 24, 16, 64, 32])

# Filters
h = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # LOW PASS
g = np.array([1/np.sqrt(2), -1/np.sqrt(2)]) # HIGH PASS

check_orthonormality(h, g)

########################################################
  
# Realizar la transformada wavelet de Haar utilizando los filtros personalizados
A1 = np.convolve(s, h, mode='valid')
D1 = np.convolve(s, g, mode='valid')

# Imprimir los resultados
print("Approximation coefficients (A1):", A1)
print("Detail coefficients (D1):", D1)

# # Definir los coeficientes de aproximación y detalle
# cA = np.array([28.28427125, 20.50609665, 48.89026304, 45.254834])
# cD = np.array([3.53553391, -3.53553391, -8.48528137, -3.53553391])

# # Definir los filtros inversos
# h_inv = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # LOW PASS inverse
# g_inv = np.array([-1/np.sqrt(2), 1/np.sqrt(2)])  # HIGH PASS inverse

# # Realizar la reconstrucción de la señal
# reconstructed_signal = np.convolve(cA, h_inv, mode='full') + np.convolve(cD, g_inv, mode='full')

# # Imprimir la señal reconstruida
# print("Señal reconstruida:", reconstructed_signal)