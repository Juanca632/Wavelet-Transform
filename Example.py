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
s = [32,32,16,8,24,16,64,32]

# Filters
h = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # LOW PASS
g = np.array([1/np.sqrt(2), -1/np.sqrt(2)]) # HIGH PASS

check_orthonormality(h, g)

########################################################
