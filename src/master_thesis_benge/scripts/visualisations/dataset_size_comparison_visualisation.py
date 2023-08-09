import matplotlib.pyplot as plt

# Assuming you have the F1 scores for each dataset size in the following list
dataset_sizes = [0, 0.2, 0.4, 0.6, 0.8, 1]
mse = [76.37, 88.18, 90.79, 91.52, 0.0, 0.0]
accuracies = [95.94, 97.87, 98.33, 98.46, 0.0, 0.0]

# Create a line plot to visualize the F1 performance
plt.figure(figsize=(10, 6))  # Increase the figure size for better visibility

# Line plots with custom colors and labels
plt.plot(dataset_sizes, f1_scores, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='F1 Score')
plt.plot(dataset_sizes, accuracies, marker='o', linestyle='-', color='darkorange', linewidth=2, label='Accuracy')

# Set plot labels and title
plt.xlabel('Dataset Size BEN-GE', fontsize=14)
plt.ylabel('Score', fontsize=14)  # Simplify ylabel since we're plotting both F1 and Accuracy
plt.title('Classification Single-Label', fontsize=16)

# Add gridlines with transparency
plt.grid(True, linestyle='--', alpha=0.7)

# Customize the appearance of tick labels
plt.xticks(dataset_sizes, ["8k", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=12)
plt.yticks(fontsize=12)

# Set the y-axis range from 0 to 100 to represent the score range
#plt.ylim(0, 100)

# Add data labels to the data points with a slight shift upwards
for x, y in zip(dataset_sizes, f1_scores):
    plt.text(x, y + 1.5, f'{y:.2f}', ha='center', va='bottom', fontsize=12, color='dodgerblue')

# Add data labels to the data points with a slight shift downwards
#for x, y in zip(dataset_sizes, accuracies):
    #plt.text(x, y - 1.5, f'{y:.2f}', ha='center', va='top', fontsize=12, color='darkorange')

# Add legend with a larger fontsize
plt.legend(loc='lower right', fontsize=12)

# Customize the appearance of the plot
plt.tight_layout()

# Show the plot
plt.show()