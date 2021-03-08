import numpy as np

#Questions on NumPy Statistics
# Compute the median of the flattened NumPy array
#median Compute the median along the specified axis.
np.median(np.array([1, 2, 3, 4, 5, 6, 7])) 

# Find Mean of a List of Numpy Array
#mean Compute the arithmetic mean along the specified axis.
np.mean(np.array([1, 2, 3]))

# Calculate the mean of array ignoring the NaN value
#nanmean Compute the arithmetic mean along the specified axis, ignoring NaNs.
np.nanmean(np.array([[20, 15, 37], [47, 13, np.nan]]))

# Get the mean value from given matrix
#matrix.mean Return mean value from given matrix
np.matrix('[64, 1; 12, 3]').mean()

# Compute the variance of the NumPy array
# var Return Variance of the array (a scalar value if axis is none)
# or array with variance values along specified axis.
np.var([20, 2, 7, 1, 34]  , dtype = np.float32)

# Compute the standard deviation of the NumPy array
# std Return tandard Deviation of the array (a scalar value if axis is none) 
# or array with standard deviation values along specified axis.
np.std([20, 2, 7, 1, 34]  , dtype = np.float32)

# Compute pearson product-moment correlation coefficients of two given NumPy arrays
# corrcoef Pearson product-moment correlation coefficients
np.corrcoef( np.array([0, 1, 2]), np.array([3, 4, 5])) 

# Calculate the mean across dimension in a 2D NumPy array
np.mean(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), axis=1)

# Calculate the average, variance and standard deviation in Python using NumPy
# average Return Default is False. If True, the tuple is returned, otherwise 
# only the average is returned
np.average([2, 4, 4, 4, 5, 5, 7, 9] )
np.var([20, 2, 7, 1, 34]  , dtype = np.float32)
np.std([20, 2, 7, 1, 34]  , dtype = np.float32)

# Describe a NumPy Array in Python
# amin takes a NumPy array as an argument and returns the minimum
# amax takes a NumPy array as an argument and returns maximum.
# ptp takes a NumPy array as an argument and returns the range of the data.
arr = np.array([4, 5, 8, 5, 6, 4, 9, 2, 4, 3, 6]) 
# measures of central tendency 
mean = np.mean(arr) 
median = np.median(arr) 
  # measures of dispersion 
np.amin(arr) 
np.amax(arr) 
np.ptp(arr) 
np.var(arr) 
np.std(arr) 