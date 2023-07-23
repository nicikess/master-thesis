import matplotlib.pyplot as plt

# Data
label_fraction = [1, 10, 50, 100]
supervised_results = [85.24, 88.58, 91.11, 92.21]
self_supervised_results = [88.59, 91.3, 92.35, 92.58]

# Plotting
plt.figure(figsize=(8, 6))

# Supervised results
plt.plot(label_fraction, supervised_results, marker='o', linestyle='-', color='blue', label='Supervised')

# Self-supervised results
plt.plot(label_fraction, self_supervised_results, marker='o', linestyle='-', color='green', label='Self-supervised')

# Title and labels
plt.title('Comparison of Supervised and Self-supervised Results \n Multi-Label World Cover Classification (Modalities: Sen-2 and Sen-1)')
plt.xlabel('Label Fraction')
plt.ylabel('F1 score')

# X-axis ticks
plt.xticks(label_fraction)
plt.gca().set_xticklabels(['1%', '10%', '50%', '100%'])

# Y-axis range
plt.ylim(80, 100)

# Legend
plt.legend()

# Grid
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()