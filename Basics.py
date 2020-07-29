import numpy as np
import time, math
# Why vectorization is preferred over general way
a = np.array([1, 2, 3, 4, 5])
print(a)
# Vectorized version
a = np.random.rand(10000000) # It returns an array of specific size
b = np.random.rand(10000000)
tic = time.time()
c = np.dot(a, b) # Calculating product of the matrix
toc = time.time()
print(c)
print("Vectorised version:", (toc-tic)*1000, "ms") # See the time difference
# Using Non - Vectorized version
c = 0
tic = time.time()
for i in range(10000000) : c+=a[i]*b[i] # Calculating product
toc = time.time()
print(c)
print("For loop:", (toc-tic)*1000, "ms") # The time difference again, vectorized version are highly efficient and takes wayy less time
# Vectorization
a = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])
print(a)
tot = a.sum(axis=0)
perc = 100*a/tot.reshape(1, 4) # Use of reshape() is encouraged to make sure you get the correct sized matrix for performing the operations
print(perc)
# Reducing bugs
a = np.random.rand(5)
print(a.shape) # It throws out a rank 1 array in python which is neither a row vector or a column vector
print(a, a.T) # The output of this code remains the same for both the cases
print(np.dot(a, a.T)) # It gives out a number rather than giving a matrix
a = np.random.rand(5, 1) # This helps us to create a specific shaped matrix as well as Tranpose and other operations work the usual way they should
print(a) # reshape() can be used to convert the dimension of the matrix to the ideal dimensions
assert(a.shape == (5, 1)) # This assert() can be used repetitively to check your code
# Comparison between math and numpy library
x = [1, 2, 3]
# print(1/1+math.exp(-x))
# you will see this give an error when you run it, because x is a vector.
print(np.exp(x)) # This gives the result smoothly
# Sigmoid Function
x = np.array([1, 2, 3])
sigmoid = 1/(1+np.exp(-x)) # The formula for calculating sigmoid function
ds = sigmoid*(1-sigmoid)
print(ds)# Formula for calculating derivative of sigmoid
# Image to Vector
image = np.array([[[ 0.67826139,  0.29380381], # This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
        [ 0.90714982,  0.52835647], # For example, in computer science, an image is represented by a 3D array of shape (length,height,depth=3). However, when you read an image as the input of an algorithm
        [ 0.4215251 ,  0.45017551]], # you convert it to a vector of shape (length∗height∗3,1). In other words, you "unroll", or reshape, the 3D array into a 1D vector.

       [[ 0.92814219,  0.96677647], # mplement image2vector() that takes an input of shape (length, height, 3) and returns a vector of shape (length*height*3, 1).
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))
print(v)
# Normalizing Rows
x = np.array([ # Normalizing leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to x/∥x∥
    [0, 3, 4],
    [1, 6, 4]])
x_norm = np.linalg.norm(x, axis = 1, keepdims = True) # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
x = x/x_norm # Divide x by its norm.
print("normalizeRows(x) = " + str(x))
# Softmax Function
x = np.array([ #softmax is a normalizing function used when your algorithm needs to classify two or more classes.
    [9, 2, 5, 0, 0], #
    [7, 5, 0, 0 ,0]])
def softmax(x):
    x_exp = np.exp(x) # Apply exp() element-wise to x. Use np.exp(...).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True) # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    s = x_exp/x_sum # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    return s #If you print the shapes of x_exp, x_sum and s above and rerun the assessment cell, you will see that x_sum is of shape (2,1) while x_exp and s are of shape (2,5)
print("softmax(x) = " + str(softmax(x))) # x_exp/x_sum works due to python broadcasting
# L1 function in Numpy
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
loss = np.sum(abs(y-yhat))
print(loss)
# L2 function in Numpy
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
loss = np.dot(y-yhat,y-yhat)
print(loss)
