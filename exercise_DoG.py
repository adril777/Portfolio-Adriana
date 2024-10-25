import numpy as np
import matplotlib.pyplot as plt

# Function for the second derivative of a Gaussian
def second_derivative_of_gaussian(x, sigma):
    return (np.exp(-(x**2/(2*sigma*sigma))) * (-(1 - x**2/(sigma*sigma)) / (np.sqrt(2*np.pi) * sigma**3)))

# This function calculates the second derivative of a one-dimensional Gaussian function at a given point x with a standard deviation sigma.

# Define the range of the independent variable 't'
t = np.linspace(-10, 10, 1000)

# Step function
def s(t):
    return np.where(t >= 0, 1, 0)

# Plot the second derivative of a Gaussian kernel with different shifts
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
for i in range(-10, 10, 5):
    plt.plot(t + i, second_derivative_of_gaussian(t, 1))
plt.plot(t, s(t))
plt.title('Step function and second derivative of Gaussian kernel')

# Plot the second derivative of a Gaussian kernel with different shifts, along with the previously defined step function. 

# Convolution with second derivative of Gaussian with sigma=1
plt.subplot(1, 3, 2)
convolution_result = np.convolve(s(t), second_derivative_of_gaussian(t, 1), mode='same')
plt.plot(t, convolution_result, 'b')
plt.title('Convolution with second derivative of Gaussian with $\sigma = 1$')

# Plot the second derivative of a Gaussian kernel with different shifts, along with the previously defined step function. 

# Convolution with second derivative of Gaussian with different sigmas
plt.subplot(1, 3, 3)
for sigma in np.linspace(0.5, 2, 3):
    convolution_result = np.convolve(s(t), second_derivative_of_gaussian(t, sigma), mode='same')
    plt.plot(t, convolution_result, 'b')
plt.title('Convolution with second derivative of Gaussian with different sigmas')

plt.show()
