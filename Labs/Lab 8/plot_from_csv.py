import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# CONSTANTS
# Add your Vref value here
TRIGGER_LEVEL = ...


def choose_csv_file():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(
        title="Select CSV File", filetypes=[("CSV Files", "*.csv")]
    )
    return file_path


def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data


def plot_data(data):
    # Extract the column name (header) to use it as the label
    column_name = data.columns[0]

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.plot(data[column_name], marker="o", linestyle="-", color="b", label=column_name)
    plt.axhline(y=TRIGGER_LEVEL, color="black", linestyle="dashed")

    # Adding titles and labels
    plt.title("Data Plot")
    plt.xlabel("Index")
    plt.ylabel(column_name)

    # Display a legend
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Open a popup to choose the CSV file
    file_path = choose_csv_file()

    # Check if a file was selected
    if file_path:
        # Read the data from the CSV
        data = read_csv(file_path)

        # Plot the data
        plot_data(data)
    else:
        print("No file selected.")
