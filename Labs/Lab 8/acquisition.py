import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import tkinter as tk
import cv2
import csv
import threading
import time
from collections import deque
from datetime import datetime
import nidaqmx
from nidaqmx.constants import (
    Edge,
    AcquisitionType,
    TerminalConfiguration,
    LineGrouping,
    READ_ALL_AVAILABLE,
)

from setup_camera import init_camera

# CONSTANTS
# Creates a subdirectory where to store catured data
SAVE_DIR = "Labs/Lab 8/data_dir"

# Specifies the amount of time before and after the trigger we capture/keep data
# Takes integer seconds for simplicity, for example, 1 gives us a total capture time of 2 seconds
#1280 x 720 @ 10 fps  uncompressed


CAMERA_BUFFER_TIME_SECONDS = 1
CAMERA_WIDTH = 1280 # Find it from video modes
CAMERA_HEIGHT = 720 # Find it from video modes
FPS = 10  # Find the maximum fps from video modes
CAMERA_INDEX = 0  # Change to correct camera with trial and error
EXPOSURE = -7
FOCUS = 1

TRIGGER_CHANNEL = "/Dev7/PFI0" # For example "/Dev2/PFI4"
DAQ_TIMEOUT = 15  # Timeout for DAQ tasks, in seconds

ANALOG_CHANNEL = "/Dev7/ai0"  # For example "/Dev2/ai0"
ANALOG_RANGE = 5  # +- value of the measurement range
ANALOG_PRETRIGGER_SAMPLES = 1000
ANALOG_SAMPLE_RATE = 10000
ANALOG_TOTAL_SAMPLES = 20000
# GLOBAL VARIABLES
# Circular buffer for camera frames
frame_buffer = deque(maxlen=FPS * CAMERA_BUFFER_TIME_SECONDS * 2)
triggered = threading.Event()  # Flag to sync threads if a trigger signal is detected
stop_flag = threading.Event()  # Flag to kill threads if a stop is issued
camera = init_camera(
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, FPS, EXPOSURE, FOCUS
)  # Camera object


def save_images(data, trigger_time=datetime.now()):
    """Save the frames around the trigger event as images."""
    image_save_dir = os.path.join(
        SAVE_DIR, f"images_{trigger_time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(image_save_dir, exist_ok=True)
    # TASK 1: implement saving images from the buffer
    for i in data:
        filename = os.path.join(image_save_dir, trigger_time.strftime('%Y%m%d_%H%M%S_%f') + f"_frame_{i[0]:.3f}.png")
        cv2.imwrite(filename, i[1])
    print(f"Images saved in {image_save_dir}")


def save_analog_data(data, trigger_time=datetime.now()):
    """Save analog data to a CSV file."""

    csv_filename = os.path.join(
        SAVE_DIR, f"analog_data_{trigger_time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    with open(csv_filename, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Voltage"])
        for voltage in data:
            csv_writer.writerow([voltage])
    print(f"Analog data saved in {csv_filename}")


def analog_capture():
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            ANALOG_CHANNEL,
            terminal_config=TerminalConfiguration.RSE,
            min_val=-ANALOG_RANGE,
            max_val=ANALOG_RANGE
        )

        task.timing.cfg_samp_clk_timing(
            rate=ANALOG_SAMPLE_RATE,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=ANALOG_TOTAL_SAMPLES
        )

        task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=TRIGGER_CHANNEL,
            trigger_edge=Edge.FALLING
        )

        task.start()
        data = task.read(number_of_samples_per_channel=ANALOG_TOTAL_SAMPLES, timeout=DAQ_TIMEOUT)
        print("All analog data acquired!")
        task.stop()
        save_analog_data(data)


def camera_capture():
    """ "Camera capture thread. Capture images and store in circular buffer."""
    while not triggered.is_set():
        ret, frame = camera.read()
        if ret:
            # Add the frame to the buffer
            frame_buffer.append((time.time(), frame))

            # Show the frame
            cv2.imshow("Live Feed", frame)

            # For exiting the program
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_flag.set()
                print("Stopping the image display...")
                break

    # After trigger, continue capturing frames for the specified time period
    if triggered.is_set() and not stop_flag.is_set():

        trigger_time = time.time()
        end_time = trigger_time + CAMERA_BUFFER_TIME_SECONDS
        while time.time() < end_time:
            ret, frame = camera.read()
            if ret:
                frame_buffer.append((time.time(), frame))

        save_images(list(frame_buffer))


def wait_for_hw_trigger():
    with nidaqmx.Task() as task:
        task.di_channels.add_di_chan("/Dev7/port0/line0", line_grouping=LineGrouping.CHAN_PER_LINE)
        task.start()

        start_time = time.time()
        last = task.read()

        seen_high = (last is True)

        while time.time() - start_time < DAQ_TIMEOUT:
            if stop_flag.is_set():
                task.stop()
                return False

            cur = task.read()

            if cur is True:
                seen_high = True

            # Only accept a falling edge if we *know* we were high at least once
            if seen_high and last is True and cur is False:
                return True

            last = cur
            time.sleep(0.0005)

    return False




def create_software_trigger_gui():
    """Create a simple GUI with a button to simulate the trigger."""
    root = tk.Tk()
    root.title("Trigger Simulation")

    # Button to trigger the event
    trigger_button = tk.Button(
        root,
        text="Simulate Trigger",
        command=triggered.set,
        bg="green",
        fg="white",
        font=("Helvetica", 16),
    )
    trigger_button.pack(pady=20)

    # Exit button
    exit_button = tk.Button(
        root,
        text="Exit",
        command=root.quit,
        bg="red",
        fg="white",
        font=("Helvetica", 16),
    )
    exit_button.pack(pady=20)

    root.mainloop()


def main():
    """Start camera and analog capture in separate threads and wait for trigger event."""
    print("Press 'q' in the camera capture window to stop the program.")

    # Make sure the output dir exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Define capture threads
    camera_thread = threading.Thread(target=camera_capture)
    analog_thread = threading.Thread(target=analog_capture)

    # Start the capture threads
    camera_thread.start()

    # UNCOMMENT THIS FOR TASK 3
    analog_thread.start()

    # Wait for the trigger, set the trigger flag

    # Create the GUI
    # You only need this for the first task, you can comment it out afterwards
    #create_software_trigger_gui()

    # UNCOMMENT THIS FOR TASK 2
    if wait_for_hw_trigger():
        print("Trigger detected!")
        triggered.set()

    # Wait for threads to finish their capture and saving
    camera_thread.join()

    # UNCOMMENT THIS FOR TASK 3
    analog_thread.join()

    # Clean up
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
