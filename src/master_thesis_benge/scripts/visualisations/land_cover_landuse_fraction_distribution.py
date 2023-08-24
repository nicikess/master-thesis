import pandas as pd
import matplotlib.pyplot as plt

def plot_fractions(normalize=False):
    # Load the CSV file into a DataFrame
    file_name = "land_cover_land_use_distribution_100_normalized.png"
    #df = pd.read_csv('/ds2/remote_sensing/ben-ge/ben-ge-s/esaworldcover/ben-ge-s_esaworldcover.csv')
    df = pd.read_csv('/netscratch2/nkesseli/master-thesis-benge/src/master_thesis_benge/scripts/data-splits/ben-ge-train.csv')

    # Display the first few rows of the dataframe to understand its structure
    df.head()

    # Drop the non-fraction columns
    fractions_df = df.drop(columns=['patch_id', 'filename'])

    # Sum up the fractions for each column
    summed_fractions = fractions_df.sum()

    # Normalize the fractions if the normalize option is set to True
    if normalize:
        summed_fractions = summed_fractions / summed_fractions.sum()

    # Set up the plot
    plt.figure(figsize=(15, 8))
    ax = summed_fractions.sort_values().plot(kind='barh', color='royalblue')

    # Set title and labels
    plt.title('Pixel Fraction Each Land Cover Type', fontsize=16)
    plt.xlabel('Pixel Fractions', fontsize=14)
    plt.ylabel('Land Cover Type', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add labels to the bars
    for i, v in enumerate(summed_fractions.sort_values()):
        ax.text(v + 0.01, i, f"{v:.2f}", color='black', fontsize=12, va='center')

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Save the figure
    save_path = "/netscratch2/nkesseli/master-thesis-benge/src/master_thesis_benge/scripts/plot_output/data_distribution/"
    plt.savefig(save_path + file_name, bbox_inches='tight', dpi=300)

plot_fractions(normalize=True)