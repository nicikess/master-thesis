import pandas as pd
import matplotlib.pyplot as plt

# Load the data -> for weather and climate zones
#data = pd.read_csv('/ds2/remote_sensing/ben-ge/ben-ge/ben-ge_meta.csv')

# Load the data -> for temperature
data = pd.read_csv('/ds2/remote_sensing/ben-ge/ben-ge/era-5/ben-ge_era-5.csv')
data['temperature_s2'] = data['temperature_s2'] - 273.15

# Set a style for the plot
plt.style.use('ggplot')

# Plotting the histogram for the modality column with improvements
plt.figure(figsize=(12,7))
plt.hist(data['temperature_s2'], bins=100, color='#3498db', edgecolor='#2980b9', alpha=0.85, rwidth=0.9)
plt.title('Distribution of Temperature S2', fontsize=18, fontweight='bold')
plt.xlabel('Temperature values', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()

# Display the plot
plt.show()

# Save the figure
save_path = "/netscratch2/nkesseli/master-thesis-benge/src/master_thesis_benge/scripts/plot_output/data_distribution/"
file_name = "temperature_distribution.png"
plt.savefig(save_path + file_name, bbox_inches='tight', dpi=300)
