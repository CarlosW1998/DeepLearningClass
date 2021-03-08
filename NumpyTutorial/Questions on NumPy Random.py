import numpy as np
import random
import numpy.matlib
#Questions on NumPy Random
#Create a Numpy array with random values
#empty Return a new array of given shape and type, without initializing entries.
np.empty([2, 2], dtype=int)

# How to choose elements from the list with different probability using NumPy?
#choice Return the numpy array of random samples.
np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)

# How to get weighted random choice in Python?
random.choices([100, 200, 300, 400, 500], weights=(10, 20, 30, 40, 50), k=5) 

# Generate Random Numbers From The Uniform Distribution using NumPy
#uniform Draw samples from a uniform distribution.
np.random.uniform(size=4) 

# Get Random Elements form geometric distribution
#geometric Return the random samples of numpy array.
np.random.geometric(0.65, 1000) 

# Get Random elements from Laplace distribution
#laplace Return the random samples as numpy array.
np.random.laplace(1.45, 15, 1000) 

# Return a Matrix of random values from a uniform distribution
#rand Return a matrix of random values with given shape.
np.matlib.rand(2, 3)

# Return a Matrix of random values from a Gaussian distribution
np.matlib.rand(2, 3)