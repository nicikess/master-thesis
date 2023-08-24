import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_data_distribution():
    folder_path = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy"
    
    # GLO
    # "/ds2/remote_sensing/ben-ge/ben-ge-s/glo-30_dem_npy"

    # S2
    #"/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy"

    # S1
    #"/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-1/s1_npy"

    # ESA world cover
    #"/ds2/remote_sensing/ben-ge/esaworldcover/npy"

    # Collect the file paths of all NumPy files in the folder
    file_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".npy")
    ]

    # Initialize histogram bins and counts
    hist_bins = np.linspace(-60, 20, 81)  # Example bin edges, adjust as needed
    hist_counts = np.zeros_like(hist_bins[:-1])

    # Iterate over each file and update the histogram counts
    for file_path in tqdm(file_paths):
        data = np.load(file_path)

        # Filter out values that are exactly 0
        #data = data[data != 0]
        #data = data[data != 1]

        counts, _ = np.histogram(data, bins=hist_bins)
        hist_counts += counts

    plt.figure(figsize=(12, 7))
    plt.bar(hist_bins[:-1], hist_counts, width=np.diff(hist_bins)[0], align="edge", color="#3498db", alpha=0.7)
    plt.xlabel("Sentinel-1 Value", fontsize=15, fontweight='bold')
    plt.ylabel("Frequency", fontsize=15, fontweight='bold')
    plt.title("Distribution of Sentinel-1 values", fontsize=18, pad=20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    # Save the figure
    save_path = "/netscratch2/nkesseli/master-thesis-benge/src/master_thesis_benge/scripts/plot_output/data_distribution/"
    file_name = "sentinel-2_data_distribution.png"
    plt.savefig(save_path + file_name, bbox_inches='tight', dpi=300)
    
    plt.show()

if __name__ == "__main__":
    visualize_data_distribution()