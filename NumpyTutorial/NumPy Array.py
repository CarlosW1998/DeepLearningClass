import numpy as np

#NumPy Array
#How to create an empty and a full NumPy array?
np.empty([2, 2], dtype=int) #Return a new array of given shape and type, without initializing entries.
np.full([2, 2], 0) #Return a new array of given shape and type, filled with fill_value.

#Create a Numpy array filled with all zeros
np.zeros((5,5), dtype=int) #Return a new array of given shape and type, filled with zeros.
#Create a Numpy array filled with all ones
np.ones((5,5), dtype=int) #Return a new array of given shape and type, filled with ones.

#Check whether a Numpy array contains a specified row
#tolist Return the array as an a.ndim-levels deep nested list of Python scalars.
[1, 1] in np.ones((2, 2)).tolist()

#How to Remove rows in Numpy array that contains non-numeric values?
#any Test whether any array element along a given axis evaluates to True.
#isnan Test element-wise for NaN and return result as a boolean array.
n_arr = np.array([[10.5, 22.5, 3.8], 
                  [41, np.nan, np.nan],
                  [45.1, 16.2, 32.7]] )
n_arr[~np.isnan(n_arr).any(axis=1)]

#Remove single-dimensional entries from the shape of an array
#squeeze Remove axes of length one from a.
np.squeeze(np.array([[[0], [1], [2]]]))

#Find the number of occurrences of a sequence in a NumPy array
#array_repr Return the string representation of an array.
np.array_repr(np.array([[2, 8, 9, 4],  
                   [9, 4, 9, 4], 
                   [4, 5, 9, 7], 
                   [2, 9, 4, 3]])).count("9, 4") 

#Find the most frequent value in a NumPy array
#bincount Count number of occurrences of each value in array of non-negative ints.
#argmax Returns the indices of the maximum values along an axis.
np.bincount(np.array([1,2,3,4,5,1,2,1,1,1]) ).argmax()

#Combining a one and a two-dimensional NumPy Array
#arrange Return evenly spaced values within a given interval.
#reshape Gives a new shape to an array without changing its data.
#nditer Efficient multi-dimensional iterator object to iterate over arrays.
for a, b in np.nditer([np.arange(5) , np.arange(10).reshape(2,5)]): 
    print("%d:%d" % (a, b),)

#How to build an array of all combinations of two NumPy arrays?
#meshgrid Return coordinate matrices from coordinate vectors.
#T The transposed array.
#One shape dimension can be -1. In this case, the value is inferred 
# from the length of the array and remaining dimensions.
np.array(np.meshgrid(np.array([1, 2]) , np.array([4, 6]) )).T.reshape(-1, 2) 

#How to add a border around a NumPy array?
#pad Pad an array.
np.pad(np.ones((2, 2)), pad_width=1, mode='constant', constant_values=0) 

#How to compare two NumPy arrays?
#all Test whether all array elements along a given axis evaluate to True.
(np.array([[1, 2], [3, 4]]) == np.array([[1, 2], [3, 4]])).all()

#How to check whether specified values are present in NumPy array?
10 in np.array([[2, 3, 0], [4, 1, 6]]) 
100 in np.array([[2, 3, 0], [4, 1, 6]]) 
2 in np.array([[2, 3, 0], [4, 1, 6]])  

#How to get all 2D diagonals of a 3D NumPy array?
#diagonal Return specified diagonals.
np.diagonal(np.arange(3 * 4 * 4).reshape(3, 4, 4),  axis1 = 1, axis2 = 2) 

#Flatten a Matrix in Python using NumPy
#flatten Return a copy of the array collapsed into one dimension.
np.array([[2, 3], [4, 5]]).flatten() 

#Flatten a 2d numpy array into 1d array
#flatten Return a copy of the array collapsed into one dimension.
np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]]).flatten() 

#Move axes of an array to new positions
#moveaxis Move axes of an array to new positions. Other axes remain in
#  their original order.
np.moveaxis(np.zeros((1, 2, 3, 4)), 0, -1)

#Interchange two axes of an array
#swapaxes Interchange two axes of an array.
np.swapaxes(np.array([[2, 4, 6]]) , 0, 1)

#NumPy – Fibonacci Series using Binet Formula
#N is the first N numbers of Fibonacci
N = 11
a = np.arange(1, N)   
# splitting of terms for easiness 
sqrtFive = np.sqrt(5) 
alpha = (1 + sqrtFive) / 2
beta = (1 - sqrtFive) / 2
# Implementation of formula 
# np.rint is used for rounding off to integer 
Fn = np.rint(((alpha ** a) - (beta ** a)) / (sqrtFive)) 
Fn

#Counts the number of non-zero values in the array
#count_nonzero Counts the number of non-zero values in the array a.
np.count_nonzero(np.array([[0, 1, 7, 0], [3, 0, 2, 19]]))

#Count the number of elements along a given axis
#size Return the number of elements along a give-n axis.
np.size(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), axis=0)

#Trim the leading and/or trailing zeros from a 1-D array
#trim_zeros Trim the leading and/or trailing zeros from a 1-D array or sequence.
np.trim_zeros(np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0)))

#Change data type of given numpy array
#astype Copy of the array, cast to a specified type.
np.array([10, 20, 30, 40, 50]).astype('float64')

#Reverse a numpy array
#flip Reverse the order of elements in an array along the given axis.
np.flip(np.array([10, 20, 30, 40, 50]), 0)

#How to make a NumPy array read-only?
#flags Information about the memory layout of the array.
#WRITEABLE can only be set True if the array owns its own memory or the 
# ultimate owner of the memory exposes a writeable buffer interface or is a string.
np.zeros(11).flags.writeable = False