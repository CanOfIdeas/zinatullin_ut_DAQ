import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory
import os

# CONSTANTS
# Make this smaller if you feel that program is running too slow. This scales the input image.
SCALE_FACTOR = 0.5
# Inverse ratio of the accumulator resolution to the image resolution in the Hough circle function.
# Increasing this number will increase performance but decerease the functions capability of detecting smaller circles.
HOUGH_CIRCLES_DP = 4


# Callback function for trackbars (not used)
def nothing(x):
    pass


def choose_image_folder():
    root = Tk()
    root.withdraw()
    folder_path = askdirectory(title="Select a Folder Containing Image Files")

    if folder_path:
        # Get a list of image files (jpg, jpeg, png) in the folder
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        return image_files
    return []


def main():
    # Load the images
    images = choose_image_folder()

    if not images:
        print("No image files found in the folder.")
        exit()

    selected_image_index = 0

    image = cv2.imread(images[selected_image_index], cv2.IMREAD_COLOR)
    image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    if image is None:
        print("Error loading image.")
        exit()

    # Create a window for the masked image
    cv2.namedWindow("Masked Image")

    # Create a window for the trackbars
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 600, 300)

    # Create trackbar for switching between images
    cv2.createTrackbar("Sel_Image", "Trackbars", 0, len(images) - 1, nothing)

    # Create HSV trackbars
    cv2.createTrackbar("U_Hue", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("L_Hue", "Trackbars", 0, 179, nothing)

    cv2.createTrackbar("U_Sat", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("L_Sat", "Trackbars", 0, 255, nothing)

    cv2.createTrackbar("U_Value", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("L_Value", "Trackbars", 0, 255, nothing)

    while True:
        # Get the current positions of the trackbars
        lower_hue = cv2.getTrackbarPos("L_Hue", "Trackbars")
        lower_saturation = cv2.getTrackbarPos("L_Sat", "Trackbars")
        lower_value = cv2.getTrackbarPos("L_Value", "Trackbars")

        upper_hue = cv2.getTrackbarPos("U_Hue", "Trackbars")
        upper_saturation = cv2.getTrackbarPos("U_Sat", "Trackbars")
        upper_value = cv2.getTrackbarPos("U_Value", "Trackbars")

        # Get the current image index from the trackbar
        new_selected_image_index = cv2.getTrackbarPos("Sel_Image", "Trackbars")

        # If the image index changes, load the new image
        if new_selected_image_index != selected_image_index:
            selected_image_index = new_selected_image_index
            image = cv2.imread(images[selected_image_index], cv2.IMREAD_COLOR)
            image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        # Convert the image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for filtering the ball
        lower_bound = np.array([lower_hue, lower_saturation, lower_value])
        upper_bound = np.array([upper_hue, upper_saturation, upper_value])

        # Create a mask using the bounds
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Convert the masked image to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Detect circles using HoughCircles, feel free to change the parameters here
        # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_CIRCLES_DP,
            minDist=1000,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=400,
        )

        # Draw circles and calculate diameter if circles are detected
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:

                x_coor = i[0]
                y_coor = i[1]
                radius = i[2]

                # Draw the diameter
                cv2.circle(masked_image, (x_coor, y_coor), radius, (0, 255, 0), 2)
                # Draw the center
                cv2.circle(masked_image, (x_coor, y_coor), 2, (0, 0, 255), 3)

                # Get the ball diameter on the input picture
                # Pay attention to the scale factor!!!
                diameter_pixels = radius * 2 / SCALE_FACTOR
                cv2.putText(
                    masked_image,
                    f"Diameter: {diameter_pixels} px",
                    (x_coor - 20, y_coor - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.putText(
                    masked_image,
                    f"Coordinates: (x:{x_coor / SCALE_FACTOR}, y:{y_coor / SCALE_FACTOR})",
                    (x_coor - 40, y_coor - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # Show the masked image with circles
        cv2.imshow("Masked Image", masked_image)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
