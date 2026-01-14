import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2


def init_camera(camera_index, width, height, fps, exposure=-3, focus=50):
    """This function assumes you are using a logitech C922 camera on Windows"""

    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)

    # We need to grab a frame here to let us configure some properties
    ret, frame = camera.read()

    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    camera.set(cv2.CAP_PROP_EXPOSURE, exposure)

    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    camera.set(cv2.CAP_PROP_FOCUS, focus)

    return camera


def main():
    pass


if __name__ == "__main__":
    main()
