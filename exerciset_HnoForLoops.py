import numpy as np

def compute_H(X, Y):
    N = X.shape[0]
    sum_x = np.sum(X)
    sum_x2 = np.sum(X**2)
    H = np.array([[sum_x2, sum_x], [sum_x, N]])
    return H

# compute_H(X, Y):
# Here, the matrix H is calculated using vectorized operations.
# H is a 2x2 matrix where the first element is the sum of x^2(i),the second and third elements are the sum of xi, and the fourth element is the number of samples or N.

def compute_b(X, Y):
    sum_yx = np.sum((Y - X) * X)
    sum_y_x = np.sum(Y - X)
    b = np.array([sum_yx, sum_y_x])
    return b

# compute_b:
# Calculate the vector b using vectorized operations.
# The first element of b is the sum of xi(yi - xi).
# The second element is the sum of (yi - xi).

# Function to generate synthetic data and test the implementation
def generate_synthetic_data(a, b, N=100, noise_std=1.0):
    X = np.random.rand(N, 1) * 10  # Values of x in the range [0, 10]
    Y = (a + 1) * X + b + np.random.normal(0, noise_std, (N, 1))  # Generate y with noise
    return X, Y

# This function generates synthetic random data to test the functions, so each time you run the code you will have a different demonstration.

# Function to estimate parameters a and b
def estimate_parameters(X, Y):
    H = compute_H(X, Y)
    b = compute_b(X, Y)
    H_inv = np.linalg.inv(H)
    params = H_inv.dot(b)
    a_estimated = params[0]
    b_estimated = params[1]
    return a_estimated, b_estimated

# In this part, the matrix H is inverted and multiplied by b to obtain the estimated parameters.

# Test with synthetic data
np.random.seed(42)
a_true = 2.0
b_true = 5.0
X, Y = generate_synthetic_data(a_true, b_true)

a_estimated, b_estimated = estimate_parameters(X, Y)

print(f"True parameters: a = {a_true}, b = {b_true}")
print(f"Estimated parameters: a = {a_estimated}, b = {b_estimated}")

# Synthetic data is generated with known parameters.
# The parameters are estimated using the defined functions and compared with the true values.
