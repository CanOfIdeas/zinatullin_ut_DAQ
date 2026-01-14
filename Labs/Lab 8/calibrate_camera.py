# Code snippets taken from: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import datetime

from setup_camera import init_camera

# CONSTANTS
# Copy them from the last tasks
CAMERA_INDEX = 0
FPS = 10  # Change to correct camera with trial and error   
EXPOSURE = -7
FOCUS = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


CHESSBOARD_SIZE = (9, 6)


def main():
    # Read these print statements for info :)
    print("===== Camera Calibration ==============")
    print("Instructions:")
    print("1. Place a chessboard pattern in front of the camera.")
    print("2. Press 'c' in the capture window to capture frames for calibration.")
    print("3. Press 'q' in the capture window to start the calibration.")
    print("=======================================")
    print("It's good to make atleast 10 to 20 different calibration images.")
    print("For each image, vary the rotation, distance and tilt of the")
    print("chessboard in relation to the camera.")
    print("Also, the image will lag if it does not find the chessboard :)")
    print("=======================================")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_size = CHESSBOARD_SIZE

    # Generate the object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(
        -1, 2
    )

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Initialize the camera and pass back a VideoCapture object
    camera = init_camera(
        CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, FPS, EXPOSURE, FOCUS
    )

    if not camera.isOpened():
        print("Error: Cannot open camera feed")
        exit()

    while True:
        # Read in a frame, ret is a boolean that says if the operation was successful
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If corners are found, refine them and display them on the frame
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)

        # Show the frame
        cv2.imshow("Camera Calibration", frame)

        # Press 'q' to exit or 'c' to capture more frames
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("c"):
            # If capture is pressed, then add the points to our list
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners2)
                print("Frame captured for calibration")
            else:
                print("No corners found")

    # When the capturing part is done, release the capture and close windows
    print("=======================================")
    print("Killing image capture")
    camera.release()
    cv2.destroyAllWindows()

    print(f"Starting calibration with {len(objpoints)} images")

    # Calibrate the camera, check opencv docs for the complicated math
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Print the calibration results
    print("=======================================")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"camera_calibration_data_{timestamp}.npz"

    # Save the calibration results for future use (it goes in the working dir)
    np.savez(
        filename,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
    )
    print("=======================================")
    print(f"Camera calibration is complete and saved as {filename}")


if __name__ == "__main__":
    main()
