import itertools
import numpy as np
import xmltodict
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Function for 3D point estimation from projections in several cameras
def estimate_3D_pt_3D_residual(pts, K, R, T):
    N = pts.shape[1]
    denominator = np.zeros((3, 3))
    numerator = np.zeros((3, 1))
    v = np.ones((3, 1))
    for j in range(N):
        v[0] = pts[0, j]
        v[1] = pts[1, j]
        v_hat = np.linalg.inv(K[j] @ R[j]) @ v
        v_hat = v_hat / np.linalg.norm(v_hat, axis=0)
        c_j = -np.transpose(R[j]) @ T[j]
        denominator += (np.eye(3) - v_hat @ np.transpose(v_hat))
        numerator += (np.eye(3) - v_hat @ np.transpose(v_hat)) @ c_j
    p = np.linalg.inv(denominator) @ numerator
    return p

# Function to reproject a 3D point into an image
def world_to_image(p, K, R, T):
    x = K @ (R @ p + T)
    x /= x[2, :]
    return x[0:2, :]

# Calibration route
calib_dir = data_dir + 'Wildtrack/calibrations/'

extrinsic_calibration_files = ['extrinsic/extr_CVLab1.xml', 'extrinsic/extr_CVLab2.xml', 'extrinsic/extr_CVLab3.xml',
                               'extrinsic/extr_CVLab4.xml', 'extrinsic/extr_IDIAP1.xml', 'extrinsic/extr_IDIAP2.xml',
                               'extrinsic/extr_IDIAP3.xml']
intrinsic_calibration_files = ['intrinsic_zero/intr_CVLab1.xml', 'intrinsic_zero/intr_CVLab2.xml', 'intrinsic_zero/intr_CVLab3.xml',
                               'intrinsic_zero/intr_CVLab4.xml', 'intrinsic_zero/intr_IDIAP1.xml', 'intrinsic_zero/intr_IDIAP2.xml',
                               'intrinsic_zero/intr_IDIAP3.xml']

R_list = []
T_list = []
K_list = []
c_list = []

# Read calibration files
for i in range(7):
    # Read extrinsic parameters
    extr_file = calib_dir + extrinsic_calibration_files[i]
    xml_data = open(extr_file, 'r').read()
    xmlDict = xmltodict.parse(xml_data)
    rot_rodriges = np.fromstring(xmlDict['opencv_storage']['rvec'], sep=' ')
    R_matrix = R.from_rotvec(rot_rodriges).as_matrix()
    R_list.append(R_matrix)
    T = np.fromstring(xmlDict['opencv_storage']['tvec'], sep=' ')
    T_list.append(T[:, None])
    c_list.append(-np.transpose(R_list[i]) @ T_list[i])

    # Read intrinsic parameters
    intr_file = calib_dir + intrinsic_calibration_files[i]
    xml_data = open(intr_file, 'r').read()
    xmlDict = xmltodict.parse(xml_data)
    K_matrix = np.fromstring(xmlDict['opencv_storage']['camera_matrix']['data'], sep=' ').reshape(3, 3)
    K_list.append(K_matrix)

# Shoe tip coordinates in images
shoe_tip = np.array([
    [831, 401], [0, 0], [967, 346], [1882, 432],
    [1650, 580], [620, 335], [765, 398]
])

# Exclude camera 2
cam_indices = [0, 2, 3, 4, 5, 6]

# Generate all possible combinations of camera pairs
cam_pairs = list(itertools.combinations(cam_indices, 2))

# List for storing errors
errors = []

# Evaluate each pair of cameras
for pair in cam_pairs:
    # Estimating the 3D point using the camera pair
    K_list_selected = [K_list[i] for i in pair]
    R_list_selected = [R_list[i] for i in pair]
    T_list_selected = [T_list[i] for i in pair]
    im_pt_selected = shoe_tip[pair, :].T

    estimated_3D_point = estimate_3D_pt_3D_residual(im_pt_selected, K_list_selected, R_list_selected, T_list_selected)

    # Reproject the estimated point and calculate error
    reprojection_error = 0.0
    for idx in pair:
        x_reprojected = world_to_image(estimated_3D_point, K_list[idx], R_list[idx], T_list[idx])
        reprojection_error += np.linalg.norm(x_reprojected.squeeze() - shoe_tip[idx, :])

    errors.append((pair, reprojection_error))

# Finding the camera pair with the smallest error
best_pair = min(errors, key=lambda x: x[1])
worst_pair = max(errors, key=lambda x: x[1])

print(f'The best pair of cameras is: {best_pair[0]} with a reprojection error: {best_pair[1]:.2f} píxeles')
print(f'The worst pair of cameras is: {worst_pair[0]} with a reprojection error:{worst_pair[1]:.2f} píxeles')
