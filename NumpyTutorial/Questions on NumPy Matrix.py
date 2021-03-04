import numpy as np
#Questions on NumPy Matrix
#Get the maximum value from given matrix
#max Return maximum value from given matrix
np.matrix('[64, 1; 12, 3]').max()

#Get the minimum value from given matrix
#min Return minimum value from given matrix
np.matrix('[64, 1; 12, 3]').min()

#Find the number of rows and columns of a given matrix using NumPy
np.arange(1,10).reshape((3, 3)).shape

#Select the elements from a given matrix
#take Return matrix of selected indexes
np.matrix('[64, 1; 12, 3]').take(3) 

#Find the sum of values in a matrix
#sum Return sum of values in a matrix
np.matrix('[64, 1; 12, 3]').sum()

#Calculate the sum of the diagonal elements of a NumPy array
#trace Return the sum along diagonals of the array.
np.array([[55, 25, 15, 41], 
            [30, 44, 2, 54], 
            [11, 45, 77, 11], 
            [11, 212, 4, 20]]).trace()

#Adding and Subtracting Matrices in Python
#add Add arguments element-wise.
#subtract Subtract arguments, element-wise.
np.add(np.array([[1, 2], [3, 4]]), np.array([[4, 5], [6, 7]]))
np.subtract(np.array([[1, 2], [3, 4]]), np.array([[4, 5], [6, 7]]))

#Ways to add row/columns in numpy array
#hstack Stack arrays in sequence horizontally (column wise).
#vstack Stack arrays in sequence vertically (row wise).
np.hstack((np.array([[1, 2], [3, 4]]), np.array([[4], [5]])))
np.vstack((np.array([[1, 2], [3, 4]]), np.array([4, 5])))

#Matrix Multiplication in NumPy
#dot Dot product of two arrays
np.dot([[1, 0], [0, 1]], [[4, 1], [2, 2]])

#Get the eigen values of a matrix
#eigvals Return the eigen values of a matrix.
np.linalg.eigvals([[1, 2], [3, 4]])

#How to Calculate the determinant of a matrix using NumPy?
#det Compute the determinant of an array.
np.linalg.det([[1, 2], [3, 4]])

#How to inverse a matrix using NumPy
#inv Returns Inverse of the matrix a.
np.linalg.inv(np.array([[6, 1, 1], 
              [4, -2, 5], 
              [2, 8, 7]]) )

#How to count the frequency of unique values in NumPy array?
#unique Find the unique elements of an array.
np.unique([1, 1, 2, 2, 3, 3])

#Multiply matrices of complex numbers using NumPy in Python
#vdot Return the dot product of two vectors.
#The vdot(a, b) function handles complex numbers differently than dot(a, b).
np.vdot(np.array([2+3j, 4+5j]) , np.array([8+7j, 5+6j])) 

#Compute the outer product of two given vectors using NumPy in Python
#outer Compute the outer product of two vectors.
np.outer(np.array([[1, 3], [2, 6]]), np.array([[0, 1], [1, 9]])) 

#Calculate inner, outer, and cross products of matrices and vectors using NumPy
#inner Inner product of two arrays.
#outer Compute the outer product of two vectors.
#cross Return the cross product of two (arrays of) vectors.
np.inner(np.array([2, 6]) , np.array([3, 10])) 
np.inner(np.array([[1, 3], [2, 6]]), np.array([[0, 1], [1, 9]])) 
np.outer(np.array([2, 6]) , np.array([3, 10])) 
np.outer(np.array([[1, 3], [2, 6]]), np.array([[0, 1], [1, 9]])) 
np.cross(np.array([2, 6]) , np.array([3, 10])) 
np.cross(np.array([[1, 3], [2, 6]]), np.array([[0, 1], [1, 9]]))

#Compute the covariance matrix of two given NumPy arrays
#cov Estimate a covariance matrix, given data and weights
np.cov(np.array([[1, 3], [2, 6]]), np.array([[0, 1], [1, 9]]))

#Convert covariance matrix to correlation matrix using Python
#corrcoef Return Pearson product-moment correlation coefficients.
x = np.array([[0.77395605, 0.43887844, 0.85859792],
       [0.69736803, 0.09417735, 0.97562235],
       [0.7611397 , 0.78606431, 0.12811363]])

np.corrcoef(x,rowvar=False)  

#Compute the Kronecker product of two mulitdimension NumPy arrays
#kron Return Kronecker product of two arrays.
np.kron(np.array([[1, 2], [3, 4]]) , np.array([[5, 6], [7, 8]])) 

#Convert the matrix into a list
#tolist Returns a new list
np.matrix('[4, 1, 12, 3]').tolist()