import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_glo_dem_30():
    folder_path = "/ds2/remote_sensing/ben-ge/ben-ge-s/glo-30_dem_npy"  # Replace with the actual path to your folder

    # Plot is only an approximation, since the ben-ge-100 is to large to plot and job is killed
    # Same issue wih ben-ge-20 -> look for fix

    # Collect the file paths of all NumPy files in the folder
    file_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".npy")
    ]

    # Iterate over each file and load the data
    all_values = []
    for file_path in file_paths:
        data = np.load(file_path)
        print(data.shape)
        input()
        all_values.extend(data.flatten())

    mean = np.mean(all_values)
    std = np.std(all_values)
    lower_threshold = mean - 2 * std  # Example lower threshold for outliers
    upper_threshold = mean + 2 * std  # Example upper threshold for outliers
    print("lower_threshold: ", lower_threshold)
    print("upper_threshold: ", upper_threshold)
    filtered_values = [
        value for value in all_values if lower_threshold <= value <= upper_threshold
    ]

    # Create a histogram of all values
    plt.hist(all_values, bins="auto")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Values")
    plt.show()

    plt.savefig(
        "/netscratch2/nkesseli/master-thesis-benge/src/master_thesis_benge/scripts/plot_output"
    )


if __name__ == "__main__":
    visualize_glo_dem_30()
