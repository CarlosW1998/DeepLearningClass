import numpy as np

#Questions on NumPy Linear Algebra
#Find a matrix or vector norm using NumPy
#norm Return Matrix or vector norm.
np.linalg.norm(np.arange(9) - 4)
np.linalg.norm((np.arange(9) - 4).reshape(3,3))

#Calculate the QR decomposition of a given matrix using NumPy
#qr Compute the qr factorization of a matrix.
np.linalg.qr(np.arange(10).reshape(2, 5))

#Compute the condition number of a given matrix using NumPy
#cond Compute the condition number of a matrix.
np.linalg.cond(np.arange(1,10).reshape(3, 3))

#Compute the eigenvalues and right eigenvectors of a given square array using NumPy?
#eig Compute the eigenvalues and right eigenvectors of a square array.
np.linalg.eig(np.arange(1,10).reshape(3, 3))

#Calculate the Euclidean distance using NumPy
np.linalg.norm(np.array([1, 2, 3]) - np.array([3, 2, 1])) 