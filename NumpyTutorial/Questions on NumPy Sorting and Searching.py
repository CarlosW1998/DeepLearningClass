import numpy as np

#Questions on NumPy Sorting and Searching
# How to get the indices of the sorted array using NumPy in Python?
#argsort Returns the indices that would sort an array.
np.argsort(np.array([3, 1, 2]))

# Finding the k smallest values of a NumPy array
#sort Return a sorted copy of an array.
k = 4
arr = np.array([23, 12, 1, 3, 4, 5, 6])
arr1 = np.sort(arr)
arr1[:k]

# How to get the n-largest values of an array using NumPy?
k = 4
arr = np.array([23, 12, 1, 3, 4, 5, 6])
arr1 = np.sort(arr)
arr1[-k:]

# Sort the values in a matrix
#sort Return a sorted matrix
i = np.matrix('[4, 1; 12, 3]')
i.sort()
i

# Filter out integers from float numpy array
#astype Copy of the array, cast to a specified type.
ini_array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0]) 
result = ini_array[ini_array != ini_array.astype(int)] 
result

# Find the indices into a sorted array
#searchsorted Find indices where elements should be inserted to maintain order.
in_arr = [2, 3, 4, 5, 6] 
np.searchsorted(in_arr, 4, side='right')