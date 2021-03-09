import numpy as np
import pandas as pd 
from PIL import Image 
from io import StringIO   

#Questions on NumPy Strings
# Repeat all the elements of a NumPy array of strings
# char.multiply Return Array of strings
np.char.multiply(np.array(['a', 'b', 'c',  
    'd', 'e'], dtype = np.str) , 3) 

# How to split the element of a given NumPy array with spaces?
# split Returns  Output Array containing of list objects.
np.char.split(np.array(['PHP C# Python C Java C++'], dtype=np.str) )

# How to insert a space between characters of all the elements of a given NumPy array?
# char.join  Return a string which is the concatenation of the strings in the sequence seq.
np.char.join(" ", ['aaa', 'bbb', 'ccc']) 

# Find the length of each string element in the Numpy array
arr = np.array(['New York', 'Lisbon', 'Beijing', 'Quebec']) 
arr_len = [len(i) for i in arr] 

# Swap the case of an array of string
# swapcase Returns  lowercased array of str or 
# unicode, depending on input type.
np.char.swapcase(np.array(['P4Q R', '4q Rp', 'Q Rp4', 'rp4q']) )

# Change the case to uppercase of elements of an array
# upper Returns uppercased array of str or unicode, 
# depending on input type.
np.char.upper(np.array(['P4Q R', '4q Rp', 'Q Rp4', 'rp4q']) )

# Change the case to lowercase of elements of an array
# lower Returns Output lowercased array of str or
# unicode, depending on input type.
np.char.lower(np.array(['P4Q R', '4q Rp', 'Q Rp4', 'rp4q']))

# Join String by a seperator
# core.defchararray.join Returns Output array of str or unicode with joined elements.
np.core.defchararray.join(['-', '*', ''], ['aaa', 'bbb', 'ccc']) 

# Check if two same shaped string arrayss one by one
# core.defchararray.not_equal Returns array of bools,
# or a single bool if arr1 and arr2 are scalars.
np.char.not_equal(np.array(['aaa', 'bbb', 'ccc']), np.array(['aaa', 'bb', 'ccc']))

# Count the number of substrings in an array
# count Returns the number of non-overlapping
# occurrences of substring sub.
np.char.count(['aaaa', 'aba', "bbba"], 'a')

# Find the lowest index of the substring in an array
# find Returns array of ints.
np.char.find(['aaaa', 'aba', "bbba"], 'a')

# Get the boolean array when values end with a particular character
# endswith Return the boolean array when values ends with a value.
np.char.endswith(['aaab', 'aba', "bbba"], 'a')

# More Questions on NumPy
# Different ways to convert a Python dictionary to a NumPy array
d = {'a': 'a', 'b': 'b', 'c': 'c'}
np.array(list(d.items()))

np.array([d.values()])

# How to convert a list and tuple into NumPy arrays?
# asarray Convert the input to an array.
np.asarray([1, 2, 3, 4, 5])
np.asarray((1, 2, 3, 4, 5))

# Ways to convert array of strings to array of floats
# fromstring A new 1-D array initialized from text data in a string.
np.array(["1.1", "1.5", "2.7", "8.9"]).astype(np.float)
np.fromstring('1.1, 1.5, 2.7, 8.9', dtype = np.float, sep =', ' ) 

# Convert a NumPy array into a csv file
data =  np.arange(1,11).reshape(2,5) 
df = pd.DataFrame(data) 
df.to_csv("dados.csv")

# How to Convert an image to NumPy array and save it to CSV file using Python?
img = Image.open('test.jpg') 
imageToMatrice = np.asarray(img)

# How to save a NumPy array to a text file?
open("file1.txt", "w+").write(str(np.array([1, 2, 3, 4, 5]) ))

# Load data from a text file
# loadtxt Returns ndarray


# Python program explaining  
# loadtxt() function 
import numpy as geek 
  
# StringIO behaves like a file object 
# loadtxt Returns ndarray
np.loadtxt(StringIO("0 1 2 \n3 4 5") ) 
  
# Plot line graph from NumPy array
x = np.arange(1, 11)  
y = x * x 
plt.title("Line graph")  
plt.xlabel("X axis")  
plt.ylabel("Y axis")  
plt.plot(x, y, color ="red")  
plt.show()

# Create Histogram using NumPy
# histogram Compute the histogram of a set of data.
a = np.random.randint(100, size =(50)) 
np.histogram(a, bins = [0, 10, 20, 30, 40, 
                        50, 60, 70, 80, 90, 
                        100]) 
  
hist, bins = np.histogram(a, bins = [0, 10,  
                                     20, 30, 
                                     40, 50, 
                                     60, 70, 
                                     80, 90, 
                                     100])  
# printing histogram 
print() 
print (hist)  
print (bins)  
print() 