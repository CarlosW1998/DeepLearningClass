import numpy as np

#Questions on NumPy Indexing
#Replace NumPy array elements that doesn’t satisfy the given condition
#where Return elements chosen from x or y depending on condition.
x = np.array([[45.42436315, 52.48558583, 10.32924763], 
                  [5.7439979, 50.58220701, 25.38213418]]) 

x[x >30] = 10
x  = np.array([[45.42436315, 52.48558583, 10.32924763], 
                  [5.7439979, 50.58220701, 25.38213418]]) 
np.where(x < 5, x, x*10)

#Return the indices of elements where the given condition is satisfied
x  = np.array([[45.42436315, 52.48558583, 10.32924763], 
                  [5.7439979, 50.58220701, 25.38213418]]) 
indieces = np.where(x < 50)

#Replace NaN values with average of columns
#nanmean Compute the arithmetic mean along the specified axis, ignoring NaNs.
x = np.array([[1.3, 2.5, 3.6, np.nan],  
            [2.6, 3.3, np.nan, 5.5], 
            [2.1, 3.2, 5.4, 6.5]])

x[np.isnan(x)] = np.nanmean(x)
x
#Replace negative value with zero in numpy array
x  = np.array([[45.42436315, 52.48558583, -10.32924763], 
                  [5.7439979, 50.58220701, -25.38213418]]) 
x[x < 0] = 0
x

#How to get values of an NumPy array at certain index positions?
#put Replaces specified elements of an array with given values.
a1 = np.array([11, 10, 22, 30, 33]) 
a2 = np.array([1, 15, 60]) 
a1.put([0, 4], a2) 
a1

#Find indices of elements equal to zero in a NumPy array
x = np.array([[1.3, 2.5, 3.6, 0],  
            [2.6, 3.3, 0, 5.5], 
            [2.1, 3.2, 5.4, 6.5]]) 
np.where(x == 0)

#How to Remove columns in Numpy array that contains non-numeric values?
x = np.array([[1.3, 2.5, 3.6, np.nan],  
            [2.6, 3.3, np.nan, 5.5], 
            [2.1, 3.2, 5.4, 6.5]])
x[:, ~np.isnan(x).any(axis=0)]

#How to access different rows of a multidimensional NumPy array?
x = np.array([[1.3, 2.5, 3.6, np.nan],  
            [2.6, 3.3, np.nan, 5.5], 
            [2.1, 3.2, 5.4, 6.5]])
x[[0, 1]]

#Get row numbers of NumPy array having element larger than X
x = np.array([[1.3, 2.5, 3.6, 0],  
            [2.6, 3.3, 0, 5.5], 
            [2.1, 3.2, 5.4, 6.5]]) 
np.where(np.any(x > 6, axis = 1))

#Get filled the diagonals of NumPy array
#fill_diagonal Return the filled value in the diagonal of an array.
x = np.array([[1.3, 2.5, 3.6, 0],  
            [2.6, 3.3, 0, 5.5], 
            [2.1, 3.2, 5.4, 6.5]]) 
np.fill_diagonal(x, 10)
x

#Check elements present in the NumPy array
#isin  Return boolean array having same size as of target_array.
x = np.array([[1.3, 2.5, 3.6, 0],  
            [2.6, 3.3, 0, 5.5], 
            [2.1, 3.2, 5.4, 6.5]]) 
np.isin(x, [1.3, 2.5, 5.4, 3.2])

#Combined array index by index
#dstack Return combined array index by index.
np.dstack((np.array([1, 2, 3]), np.array([4, 5, 6])))
