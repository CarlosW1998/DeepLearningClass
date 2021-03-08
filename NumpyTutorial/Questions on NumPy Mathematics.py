import numpy as np

#Questions on NumPy Mathematics
# How to get element-wise true division of an array using Numpy?
#true_divide Returns a true division of the inputs, element-wise.
np.true_divide(np.arange(5), 4)

# How to calculate the element-wise absolute value of NumPy array?
#absolute Calculate the absolute value element-wise.
np.absolute(np.array([-1, -2, -3]))

# Compute the negative of the NumPy array
#negative Numerical negative, element-wise.
np.negative([1.,-1.])

# Multiply 2d numpy array corresponding to 1d array
(np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]]).T * np.array([0, 2, 3]) ).T 

# Computes the inner product of two arrays
#inner Inner product of two arrays.
np.inner(np.array([[1, 4], [5, 6]]), np.array([[2, 4], [5, 2]]))

# Compute the nth percentile of the NumPy array
#percentile Compute the q-th percentile of the data along the specified axis.
np.percentile(np.array([[10, 7, 4], [3, 2, 1]]), 50)

# Calculate the n-th order discrete difference along the given axis
#diff Calculate the n-th discrete difference along the given axis.
np.diff(np.array([1, 2, 4, 7, 0]))

# Calculate the sum of all columns in a 2D NumPy array
#sum Sum of array elements over a given axis.
np.sum([[1.2, 2.3], [3.4, 4.5]] , axis = 0)

# Calculate average values of two given NumPy arrays
arr1 = np.array([[3, 4], [8, 2]]) 
arr2 = np.array([[1, 0], [6, 6]]) 
avg = (arr1 + arr2) / 2
avg

# How to compute numerical negative value for all elements in a given NumPy array?
np.negative(np.array([-1, -2, -3,  1, 2, 3, 0])) 

# How to get the floor, ceiling and truncated values of the elements of a numpy array?
#floor Return the floor of the input, element-wise.
#ceil Return the ceiling of the input, element-wise.
#trunc The truncated of each element, with float data-type
np.floor(np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])) 
np.ceil(np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])) 
np.trunc(np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])) 

# How to round elements of the NumPy array to the nearest integer?
# rint Round elements of the array to the nearest integer.
np.rint(np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])) 

# Find the round off the values of the given matrix
#matrix.round  Return rounded values in matrix
np.matrix('[6.4, 1.3; 12.7, 32.3]').round()

# Determine the positive square-root of an array
#sqrt Returns the square root of the number in an array.
np.sqrt([1, 4, 9, 16]) 

# Evaluate Einstein’s summation convention of two multidimensional NumPy arrays
#einsum Evaluates the Einstein summation convention on the operands.
np.einsum("mk,kn", np.array([[1, 2], [0, 2]]) , np.array([[0, 1], [3, 4]]) ) 