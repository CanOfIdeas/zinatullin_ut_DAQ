# Code snippets taken from:https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2
import os

# CONSTANTS
# Calibration data you got from the calibration code
CALIBRATION_DATA = 'camera_calibration_data_20251219_181808_201615.npz'
# Folder containing the input images
IMAGE_FOLDER = r"Labs/Lab 8/data_dir/images_20251219_181041"
# Folder to save the undistorted images to
OUTPUT_FOLDER = r"Labs/Lab 8/data_dir/images_20251219_181041_undistorted"


def undistort_image(img, camera_matrix, dist_coeffs):
    """Takes an image and calibration data as inputs and outputs an undistorted image"""
    # Implement this function using the documentation
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, None)
    return undistorted_img


def main():
    """Loads in the calibration data and runs undistortion on all images in the input directory"""

    # Make sure the output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load the camera calibration data from the .npz file
    calibration_data = np.load(CALIBRATION_DATA)

    # Extract the camera matrix and distortion coefficients
    camera_matrix = calibration_data["camera_matrix"]
    dist_coeffs = calibration_data["dist_coeffs"]

    print(f"Camera Matrix:\n{camera_matrix}")
    print(f"Distortion Coefficients:\n{dist_coeffs}")

    # Go through all image files in specified folder and convert them to undistorted images
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith((".png", ".jpg", ".jpeg")):

            # Get the path of the image and read it in
            img_path = os.path.join(IMAGE_FOLDER, filename)
            img = cv2.imread(img_path)

            if img is not None:
                # Undistort the image and save it
                undistorted_img = undistort_image(img, camera_matrix, dist_coeffs)
                output_path = os.path.join(OUTPUT_FOLDER, f"undistorted_{filename}")
                cv2.imwrite(output_path, undistorted_img)
                print(f"Undistorted image saved: {output_path}")


if __name__ == "__main__":
    main()
