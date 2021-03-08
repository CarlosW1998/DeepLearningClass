import numpy as np

#Questions on Polynomial
# Define a polynomial function
#poly1d Return Polynomial and the operation applied
np.poly1d([4, 9, 5, 4])

# How to add one polynomial to another using NumPy in Python?
#polyadd Find the sum of two polynomials.
# p(x) = 5(x**2) + (-2)x +5  == (5,-2,5) 
# q(x) = 2(x**2) + (-5)x +2  == (2,-5,2) 
np.polynomial.polynomial.polyadd((5,-2,5) ,(2,-5,2) ) 

# How to subtract one polynomial to another using NumPy in Python?
#polysub Difference (subtraction) of two polynomials.
# p(x) = 5(x**2) + (-2)x +5  == (5,-2,5) 
# q(x) = 2(x**2) + (-5)x +2  == (2,-5,2) 
np.polynomial.polynomial.polysub((5,-2,5) ,(2,-5,2)) 

# How to multiply a polynomial to another using NumPy in Python?
#polymul Find the product of two polynomials.
# p(x) = 5(x**2) + (-2)x +5  == (5,-2,5) 
# q(x) = 2(x**2) + (-5)x +2  == (2,-5,2) 
np.polynomial.polynomial.polymul((5,-2,5) ,(2,-5,2)) 

# How to divide a polynomial to another using NumPy in Python?
#polydiv Returns the quotient and remainder of polynomial division.
# p(x) = 5(x**2) + (-2)x +5  == (5,-2,5) 
# q(x) = 2(x**2) + (-5)x +2  == (2,-5,2) 
np.polynomial.polynomial.polydiv((5,-2,5) ,(2,-5,2)) 

# Find the roots of the polynomials using NumPy
#roots Return an array containing the roots of the polynomial.
np.roots( [1, 2, 1] )

# Evaluate a 2-D polynomial series on the Cartesian product
# polygrid2d Return  The values of the two dimensional polynomial
# series at points in the Cartesian product of x and y.
np.polynomial.polynomial.polygrid2d([7, 9], [8, 10], 
    np.array([[1, 3, 5], [2, 4, 6]])) 

# Evaluate a 3-D polynomial series on the Cartesian product
# polygrid3d Return The values of the two dimensional polynomial 
# series at points in the Cartesian product of x and y.
np.polynomial.polynomial.polygrid3d([7, 9], [8, 10], [5, 6], 
    np.array([[1, 3, 5], [2, 4, 6], [10, 11, 12]])) 