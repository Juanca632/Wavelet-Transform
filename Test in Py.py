import numpy as np
import cv2

root2 = np.sqrt(2)

def split_matrix(matrix): # Split the matrix into 4 quadrants
    leftSide, rightSide = np.split(matrix, 2, axis=1)
    appr, hD = np.split(leftSide, 2, axis=0)
    vD, dD = np.split(rightSide, 2, axis=0)
    
    return appr, vD, hD, dD

def direct_wavelet2D_H(matrix): # Does the direct wavelet transform horizontally
    Nf,N2,output = matrix.shape[0], matrix.shape[0]//2 ,np.zeros_like(matrix,dtype=float)
    for c in range(0, Nf, 2):
        output[:, c//2] = (matrix[:, c] + matrix[:, c+1]) / root2 
        output[:, c//2 + N2] = (matrix[:, c] - matrix[:, c+1]) / root2
    return output

def direct_wavelet2D_V(matrix): # Does the direct wavelet transform vertically
    Nf,N2,output = matrix.shape[0], matrix.shape[0]//2, np.copy(matrix)
    for f in range(0, Nf, 2):
        output[f // 2, :N2] = (matrix[f, :N2] + matrix[f + 1, :N2]) / root2
        output[f//2 + N2, :N2] = (matrix[f, :N2] - matrix[f + 1, :N2]) / root2
    return output

def p_new(matrix): # Normalize the matrix
    max,min  = np.max(matrix), np.min(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = ((matrix[i][j]-min)*255.0)/(max-min)
    return matrix

imgOr = cv2.imread((r'C:\Users\ZIGH\Documents\ENSEA\INTERNSHIP\Wavelet-Transform\image_color.jpg'), cv2.IMREAD_GRAYSCALE)
imgOr = np.array(imgOr)


# DIRECT WAVELET TRANSFORM 
result_data = direct_wavelet2D_H(imgOr)
leftSide, rightSide = np.split(result_data, 2, axis=1)
leftSide, rightSide = p_new(leftSide), p_new(rightSide)
horizontal1 = np.concatenate((leftSide, rightSide), axis=1)
horizontal1 = direct_wavelet2D_V(horizontal1)
cA, cV, cH, cD = split_matrix(horizontal1)
cA = p_new(cA)
cV = p_new(cV)
cH = p_new(cH)
cD = p_new(cD)


cv2.imshow("Image", cA)

cv2.waitKey(0)
cv2.destroyAllWindows()