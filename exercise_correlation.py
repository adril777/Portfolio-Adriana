import numpy as np

# cross-correlation of a complementary image (1 - f) with a filter h using two different approaches to handle out-of-bounds indices.

h = np.array([[-1, 1], [-1, 1]])
f = np.zeros((10, 10))
f[3:8, 3:8] = np.ones((5, 5))  # block assignment of values

print("Image f=\n", f, "\nFilter/Kernel h=\n", h)

# Original function my_array_ref
# This function returns the value at position (i, j) of the matrix 'a' if the indices are within bounds.
# If the indices are out of bounds, it returns 0.

# Original my_array_ref function
def my_array_ref(a, i, j):
    if (0 <= i < a.shape[0]) and (0 <= j < a.shape[1]):
        return a[i, j]
    else:
        return 0

# Compute (1 - f) ⊗ h with original my_array_ref
g1 = np.zeros((10, 10))  # output array
f_complement = 1 - f

# Perform the cross-correlation

for i in range(f.shape[0]):
    for j in range(f.shape[1]):
        prod_sum = 0.0
        for k in range(h.shape[0]):
            for l in range(h.shape[1]):
                prod_sum += my_array_ref(h, k, l) * my_array_ref(f_complement, i + k, j + l)
        g1[i, j] = prod_sum

print("Output of (1 - f) ⊗ h with original my_array_ref =\n", g1)

# Modified my_array_ref function to return boundary value
def my_array_ref(a, i, j):
    if i < 0:
        i = 0
    elif i >= a.shape[0]:
        i = a.shape[0] - 1
    if j < 0:
        j = 0
    elif j >= a.shape[1]:
        j = a.shape[1] - 1
    return a[i, j]

# Re-compute (1 - f) ⊗ h with modified my_array_ref
g2 = np.zeros((10, 10))  # output array

# Perform the cross-correlation with the modified function
for i in range(f.shape[0]):
    for j in range(f.shape[1]):
        prod_sum = 0.0
        for k in range(h.shape[0]):
            for l in range(h.shape[1]):
                prod_sum += my_array_ref(h, k, l) * my_array_ref(f_complement, i + k, j + l)
        g2[i, j] = prod_sum

print("Output of (1 - f) ⊗ h with modified my_array_ref =\n", g2)

# Compare the outputs
difference = g1 - g2
print("Difference between the outputs =\n", difference)
