import numpy as np
import cv2
import IPython
from PIL import Image
from scipy.sparse import csr_matrix


def split_matrix(mat):  # divide the matrix into 4 quadrants
    h, w = mat.shape
    return mat[:h // 2, :w // 2], mat[:h // 2, w // 2:], mat[h // 2:, :w // 2], mat[h // 2:, w // 2:]


def concatenate_matrices(m1, m2, m3, m4):
    return np.concatenate((np.concatenate((m1, m2), axis=1),
                           np.concatenate((m3, m4), axis=1)), axis=0)


def imshow(image, width=None):
    _, ret = cv2.imencode('.jpg', image)
    i = IPython.display.Image(data=ret, width=width)
    IPython.display.display(i)


def jpg_to_pgm(jpg_path):
    image = Image.open(jpg_path)
    image_gray = image.convert('L')
    width = image_gray.size[0]

    if (width > 192):
        image_gray = image_gray.resize((192, 192))
        pixels = list(image_gray.getdata())

    else:
        pixels = list(image_gray.getdata())

    return np.array(pixels).reshape((192, 192))


def new_p(matrix):
    max, min = np.max(matrix), np.min(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = ((matrix[i][j]-min)*255.0)/(max-min)
    return matrix


def CSF_vieja(image, h_coeffs, g_coeffs):
    rows, cols = image.shape
    matrix_coeffs = np.zeros((rows, cols))

    for i in range(rows // 2):
        if i < rows // 2 - 1:
            matrix_coeffs[i, 2 * i:2 * i + 4] = h_coeffs
        else:
            matrix_coeffs[i, 2 * i:2 * i + 2] = h_coeffs[:2]
            matrix_coeffs[i, 0:2] = h_coeffs[2:]

    for i in range(rows // 2):
        if i < rows // 2 - 1:
            matrix_coeffs[i + rows // 2, 2 * i:2 * i + 4] = g_coeffs
        else:
            matrix_coeffs[i + rows // 2, 2 * i:2 * i + 2] = g_coeffs[:2]
            matrix_coeffs[i + rows // 2, 0:2] = g_coeffs[2:]

    return csr_matrix(matrix_coeffs)


################################
def function_daubechies_coeffs(image, h_coeffs, g_coeffs):
    coeffs_h, coeffs_g, matrix_coeffs = h_coeffs.copy(
    ), g_coeffs.copy(), np.zeros(image.shape)

    zeros_array = np.zeros(abs(matrix_coeffs.shape[0]-coeffs_h.shape[0]))
    if matrix_coeffs.shape[0] >= coeffs_h.shape[0]:
        coeffs_h = np.append(coeffs_h, zeros_array)
        coeffs_g = np.append(coeffs_g, zeros_array)
    else:
        coeffs_h = coeffs_h[:-zeros_array.shape[0]]
        coeffs_g = coeffs_g[:-zeros_array.shape[0]]

    for i in range(matrix_coeffs.shape[0]):
        if i < matrix_coeffs.shape[0]//2:
            matrix_coeffs[i, :] = np.roll(coeffs_h, i*2)
        else:
            matrix_coeffs[i, :] = np.roll(coeffs_g, i*2)

    return csr_matrix(matrix_coeffs)


################################
def daubechies4_wavelet_transform_2D(daubechies_coeffs, image, scale, width, h_coeffs, g_coeffs, details=None):
    result_H = []

    for row in image:  # ESTA PARTE HACE LA HORIZONTAL
        result_H.append(daubechies_coeffs.dot(row))
    result_V = daubechies_coeffs.dot(result_H)

    if scale == 1:
        normalized = result_V.copy()
        aP, cH, cV, cD = split_matrix(normalized)
        normalized = concatenate_matrices(
            new_p(aP), new_p(cH), new_p(cV), new_p(cD))
        if details is not None:
            for i in range(len(details)//3):
                result_V = concatenate_matrices(
                    result_V, details[0+(i*3)], details[1+(i*3)], details[2+(i*3)])
                normalized = concatenate_matrices(new_p(normalized), new_p(
                    details[0+(i*3)]), new_p(details[1+(i*3)]), new_p(details[2+(i*3)]))

            return result_V, normalized
        else:
            return result_V, normalized

    elif (np.log2(width//4) >= scale > 1):
        aP, cH, cV, cD = split_matrix(result_V)
        details = ([cH, cV, cD] +
                   details) if details is not None else [cH, cV, cD]
        return daubechies4_wavelet_transform_2D(function_daubechies_coeffs(aP, h_coeffs, g_coeffs), aP, scale-1, aP.shape[0], h_coeffs, g_coeffs, details)

    else:
        print(
            f"Error. The scale value has to be between 1 - {int(np.log2(width//4))}")
    return np.ones_like(image), np.ones_like(image)


################################
def inverse_daubechies4_wavelet_transform_2D(daubechies_coeffs, transform_result, scale, width, h_coeffs, g_coeffs):
    new_transform = transform_result.copy()
    image = []

    if scale == 1:
        daubechies_coeffs_transpose = daubechies_coeffs.T

        result_H = daubechies_coeffs_transpose.dot(transform_result)
        for row in result_H:
            original_row = daubechies_coeffs_transpose.dot(row)
            image.append(original_row)

        return np.array(image)

    else:
        tam_img = width // (2 ** scale) * 2
        aP = new_transform[:tam_img, :tam_img]
        daubechies_coeffs_transpose = function_daubechies_coeffs(
            aP, h_coeffs, g_coeffs).T

        result_H = daubechies_coeffs_transpose.dot(aP)
        for row in result_H:
            original_row = daubechies_coeffs_transpose.dot(row)
            image.append(original_row)

        new_transform[:len(image), :len(image)] = image

        return inverse_daubechies4_wavelet_transform_2D(function_daubechies_coeffs(new_transform, h_coeffs, g_coeffs), new_transform, scale-1, new_transform.shape[0], h_coeffs, g_coeffs)


################################
def zeros(image, scale, black_frame="appr"):
    tam_img = image.shape[0] // (2 ** scale)

    if black_frame == "dH":
        image[:tam_img, tam_img:2*tam_img] = np.zeros((tam_img, tam_img))
    elif black_frame == "dV":
        image[tam_img:2*tam_img, :tam_img] = np.zeros((tam_img, tam_img))
    elif black_frame == "dD":
        image[tam_img:2*tam_img, tam_img:2 *
              tam_img] = np.zeros((tam_img, tam_img))
    else:
        image[:tam_img, :tam_img] = np.zeros((tam_img, tam_img))
    return image


def verificar_ortonormalidad(v1, v2):
    if len(v1) != len(v2):
        print("The vectors are not similar, they are not orthonormal")

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if np.isclose(dot_product, 0) and np.isclose(norm_v1, 1) and np.isclose(norm_v2, 1):
        print(f"The vectors are orthonormal, the dot product is: {dot_product} and the norm of each vector is {norm_v1, norm_v2} respectively")
    else:
        print("The vectors are similar but they are not orthonormal")


def get_details(scale, result_img, normalized):
    result_img1 = result_img.copy()
    normalized1 = normalized.copy()
    result_img2 = result_img.copy()
    normalized2 = normalized.copy()
    result_img3 = result_img.copy()
    normalized3 = normalized.copy()

    result_img1 = zeros(result_img1, scale, black_frame="appr")
    normalized1 = zeros(normalized1, scale, black_frame="appr")
    result_img2 = zeros(result_img2, scale, black_frame="appr")
    normalized2 = zeros(normalized2, scale, black_frame="appr")
    result_img3 = zeros(result_img3, scale, black_frame="appr")
    normalized3 = zeros(normalized3, scale, black_frame="appr")


    scale_aux = scale
    while scale_aux >= 1:

        ################### ZERO ADDING ####### dH #############
        result_img1 = zeros(result_img1, scale_aux, black_frame="dV")
        result_img1 = zeros(result_img1, scale_aux, black_frame="dD")

        normalized1 = zeros(normalized1, scale_aux, black_frame="dV")
        normalized1 = zeros(normalized1, scale_aux, black_frame="dD")

        ################### ZERO ADDING ####### dV #############
        result_img2 = zeros(result_img2, scale_aux, black_frame="dH")
        result_img2 = zeros(result_img2, scale_aux, black_frame="dD")

        normalized2 = zeros(normalized2, scale_aux, black_frame="dH")
        normalized2 = zeros(normalized2, scale_aux, black_frame="dD")

        ################### ZERO ADDING ####### dD #############
        result_img3 = zeros(result_img3, scale_aux, black_frame="dH")
        result_img3 = zeros(result_img3, scale_aux, black_frame="dV")

        normalized3 = zeros(normalized3, scale_aux, black_frame="dH")
        normalized3 = zeros(normalized3, scale_aux, black_frame="dV")

        scale_aux -= 1
    return result_img1, normalized1, result_img2, normalized2, result_img3, normalized3


if __name__ == '__main__':

    ROOT2, ROOT3 = np.sqrt(2), np.sqrt(3)

    H_COEFFS = np.array([(1 + ROOT3) / (4 * ROOT2), (3 + ROOT3) /
                        (4 * ROOT2), (3 - ROOT3) / (4 * ROOT2), (1 - ROOT3) / (4 * ROOT2)])
    G_COEFFS = np.array([H_COEFFS[3], -H_COEFFS[2], H_COEFFS[1], -H_COEFFS[0]])

    hola = function_daubechies_coeffs(
        np.random.randint(0.0, 52.0, (8, 8)), H_COEFFS, G_COEFFS)
    print(hola)
