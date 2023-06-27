import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio

def save_histogram_and_values(folder_path, histogram_filename, values_filename):
    # Collect the file paths of all NumPy files in the folder
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.tif')]

    # Iterate over each file and load the data
    all_values = []
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Assuming single-band TIFFs
            all_values.extend(data.flatten())

    # Calculate the histogram values and bin edges
    histogram_values, bin_edges = np.histogram(all_values, bins='auto')

    # Calculate the histogram counts
    histogram_counts = np.diff(bin_edges)

    # Save the histogram values and counts to a file
    np.savetxt(histogram_filename, np.vstack((histogram_values, histogram_counts)).T, delimiter=',')

    # Calculate the minimum and maximum of all values
    min_value = np.min(all_values)
    max_value = np.max(all_values)

    # Save the min and max values to a separate file
    with open(values_filename, 'w') as f:
        f.write(f"Minimum value: {min_value}\n")
        f.write(f"Maximum value: {max_value}\n")

    return min_value, max_value

# Example usage
folder_path = '/ds2/remote_sensing/ben-ge/ben-ge/glo-30_dem'  # Replace with the actual path to your folder
histogram_filename = "plot_output/glo_histogram_values.txt"
values_filename = "plot_output/glo_min_max_values.txt"

min_value, max_value = save_histogram_and_values(folder_path, histogram_filename, values_filename)
