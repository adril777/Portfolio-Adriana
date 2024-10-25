def padded_add(image1, image2):
    """
    This function should handle dimension mismatches
    and pad the smaller image to match the dimensions of the larger image.
    """
    import numpy as np
    
    # Determine the size of the result image
    rows = max(image1.shape[0], image2.shape[0])
    cols = max(image1.shape[1], image2.shape[1])
    
    # Create empty arrays with the target size
    result1 = np.zeros((rows, cols))
    result2 = np.zeros((rows, cols))
    
    # Fill the result arrays with the input images
    result1[:image1.shape[0], :image1.shape[1]] = image1
    result2[:image2.shape[0], :image2.shape[1]] = image2
    
    # Return the sum of the two padded images
    return result1 + result2

def create_difference_pyramid(up_pyramid, down_pyramid):
    """
    Creates a difference pyramid by calculating the difference between corresponding
    levels of the up and down pyramids.
    """
    # Check if the up and down pyramids have the same number of levels
    if len(up_pyramid) != len(down_pyramid):
        print('Error -- the up and down pyramids are of different heights')
        return 0  # Return 0 to indicate an error

    nlevels = len(up_pyramid)  # Number of levels in the pyramids
    difference_pyramid = []  # Initialize the difference pyramid list

    # Append the top-most level of the up pyramid directly to the difference pyramid (no difference calculation needed)
    difference_pyramid.append(up_pyramid[-1])

    # Calculate the difference for each level in the pyramids, starting from the smallest level
    for i in range(1, nlevels):
        # Compute the difference between the corresponding levels of the up and down pyramids
        diff_im = padded_add(up_pyramid[-(i+1)], -down_pyramid[i])

        # Append the difference image to the difference pyramid
        difference_pyramid.append(diff_im)

    return difference_pyramid  # Return the constructed difference pyramid


