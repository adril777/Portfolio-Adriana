import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

# The SIFT Algorithm
# In this case, similarities between photos of a Coca-Cola Santa Claus figure are studied.
# In the first version of OpenCV, these points were more visible. In version 2, only the noise and the connecting lines between these points are shown.

# Path to images
data_dir = 'images/'
image_files = glob.glob(data_dir + '*.jpg')  # Search JPG Files

# Upload the images
images = [cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) for image_file in image_files]

# Create the SIFT detector
sift = cv2.SIFT_create()

# Detect key points and descriptors in each image
keypoints = []
descriptors = []
for image in images:
    kp, des = sift.detectAndCompute(image, None)
    keypoints.append(kp)
    descriptors.append(des)

# Create the BFMatcher matchmaker
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match features between all image combinations
matches = []
for i in range(len(images)):
    for j in range(i+1, len(images)):
        match = bf.match(descriptors[i], descriptors[j])
        matches.append((i, j, sorted(match, key=lambda x: x.distance)))

# The features are matched between all possible combinations of images.

# Calculate a quantitative measure of match quality
match_quality = []
for match in matches:
    i, j, m = match
    good_matches = [m_ for m_ in m if m_.distance < 0.75 * max(len(descriptors[i]), len(descriptors[j]))]
    quality = len(good_matches) / max(len(descriptors[i]), len(descriptors[j]))
    match_quality.append((i, j, quality))

# The quality of the matching is calculated as the ratio of good matches to the maximum number of descriptors between two images.

# Print the qualities of the pairings
for mq in match_quality:
    i, j, quality = mq
    print(f"Match quality between image {i} and image {j}: {quality:.2f}")

# Draw the detected key points for visualization purposes
sift_images = [cv2.drawKeypoints(images[i], keypoints[i], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) for i in range(len(images))]

# Draw pairings for the first image combination as an example
img1_idx, img2_idx, m = matches[0]
match_image = cv2.drawMatches(images[img1_idx], keypoints[img1_idx], images[img2_idx], keypoints[img2_idx], m[:20], None, flags=2)

# cv2.drawKeypoints and cv2.drawMatches are used to visualize the key points and the matches.
# matplotlib is used to display the original images, the detected features, the descriptors, and the matches.

#----------------------------------Display-------------------------------------------------------
fig, ax = plt.subplots(nrows=4, ncols=2)
fig.set_size_inches(15, 20)

# Show the first two original images
ax[0,0].imshow(images[img1_idx], 'gray')
ax[0,1].imshow(images[img2_idx], 'gray')

# Show detected SIFT features
ax[1,0].imshow(sift_images[img1_idx])
ax[1,0].set_title('SIFT features (Image 1)')
ax[1,1].imshow(sift_images[img2_idx])
ax[1,1].set_title('SIFT features (Image 2)')

# Show SIFT descriptors
ax[2,0].imshow(descriptors[img1_idx].transpose(), aspect='auto')
ax[2,0].set_title('SIFT descriptors (Image 1)')
ax[2,0].set_xlabel('SIFT points')
ax[2,0].set_ylabel('128 dim descriptor')
ax[2,1].imshow(descriptors[img2_idx].transpose(), aspect='auto')
ax[2,1].set_title('SIFT descriptors (Image 2)')
ax[2,1].set_xlabel('SIFT points')
ax[2,1].set_ylabel('128 dim descriptor')

# Show pairings
ax[3,0].imshow(match_image)
ax[3,0].set_title('Feature Matches')

plt.tight_layout()
plt.show()

# Save matchup results
for i, (img1_idx, img2_idx, m) in enumerate(matches):
    match_img = cv2.drawMatches(images[img1_idx], keypoints[img1_idx], images[img2_idx], keypoints[img2_idx], m[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(f'match_{img1_idx}_{img2_idx}.png', match_img)

cv2.destroyAllWindows()
