import numpy as np
import pandas as pd
import tqdm as tqdm
import matplotlib.pyplot as plt

class ExploratoryDataAnalysis:

    def __init__(
        self,
        data_set,
        esaworldcover_index,
        file_path_era5,
        file_path_sentinel_1_2_metadata,
    ):
        self.data_set = data_set
        self.esaworldcover_index = esaworldcover_index
        self.file_path_era5 = file_path_era5
        self.file_path_sentinel_1_2_metadata = file_path_sentinel_1_2_metadata

    def distribution_barchart(self, modality="s2_img"):
        dataset_arrays = []
        for i in tqdm(range(len(self.data_set))):
            img = self.data_set.__getitem__(i)[modality]
            img_rounded = np.round(img, 0)
            dataset_arrays.append(img_rounded)
        combined_array = np.concatenate(dataset_arrays).ravel()
        unique_values, counts = np.unique(combined_array, return_counts=True)

        # Remove second value, to remove the 1 values
        # unique_values = np.delete(unique_values, 1)
        # counts = np.delete(counts, 1)

        # Make plot
        plt.bar(unique_values, counts)
        plt.title("Histogram of unique values")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.ticklabel_format(style="plain")
        plt.show()

    def boxplot(self, modality="s2_img"):
        dataset_arrays = []
        for i in tqdm(range(len(self.data_set))):
            img = self.data_set.__getitem__(i)[modality]
            img_rounded = np.round(img, 0)
            dataset_arrays.append(img_rounded)
        combined_array = np.concatenate(dataset_arrays).ravel()
        unique_values, counts = np.unique(combined_array, return_counts=True)

        # Remove second value (at index 1), to remove all the 1 values
        # counts = np.delete(counts, 1)

        # Create figure and axis objects
        fig, ax = plt.subplots()
        # Plot the boxplot
        ax.boxplot(combined_array, showfliers=True, flierprops=dict(markersize=0.3))
        # Set title and axis labels
        ax.set_title("Boxplot")
        ax.set_xlabel("Data")
        ax.set_ylabel("Values")
        # Show the plot
        plt.show()

    def describe_modality(self, modality="s2_img"):
        dataset_arrays = []
        for i in tqdm(range(len(self))):
            img = self.__getitem__(i)[modality]
            img_rounded = np.round(img, 0)
            dataset_arrays.append(img_rounded)
        combined_array = np.concatenate(dataset_arrays).ravel()

        description = pd.DataFrame(combined_array).describe()
        print(description)

    def esaworldcover_statistics(self):
        # Select column from 2 until the end
        df = self.esaworldcover_index.iloc[:, 2:len(self.esaworldcover_index.columns)]
        # Get the sum of each column
        df_sum = df.sum(axis=0)
        fig, ax = plt.subplots(figsize=(15, 10))
        # create the pie chart and adjust label distance
        ax.pie(df_sum, labels=df_sum.index, autopct='%1.1f%%', labeldistance=1.3, pctdistance=0.9)

        # create a legend and set the font size
        ax.legend(loc='lower left', bbox_to_anchor=(-0.5, -0.1), fontsize=12)

        # set the title
        ax.set_title('Culumn sum', fontsize=16)

        # display the plot
        plt.show()

    def era5_statistics(self):
        df = pd.read_csv(self.file_path_era5)
        df_numerical = df.drop(columns=['patch_id', 'patch_id_s1'])
        df_description = df_numerical.describe()
        print(df_description.to_string())

    def era5_statistics_plot(self):
        df = pd.read_csv(self.file_path_era5)
        df_numerical = df['atmpressure_level']
        # Convert to ndarray
        df_numerical = df_numerical.to_numpy()
        df_numerical_rounded = np.round(df_numerical, 1)

        unique_values, counts = np.unique(df_numerical_rounded, return_counts=True)

        # Remove second value, to remove the 1 values
        # unique_values = np.delete(unique_values, 1)
        # counts = np.delete(counts, 1)

        # Make plot
        plt.bar(unique_values, counts)
        plt.title("Histogram of unique values")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.ticklabel_format(style="plain")
        plt.show()
        print(f'Unique values:  "{unique_values}')
        print(f'Count:          "{counts}')

    def sentinel_1_2_metadata_statistics(self):
        df = pd.read_csv(self.file_path_sentinel_1_2_metadata)
        print(df.timestamp_s2.dtype)
        df['timestamp_s1'] = pd.to_datetime(df['timestamp_s1'])
        df['timestamp_s2'] = pd.to_datetime(df['timestamp_s2'])
        print(df.timestamp_s2.dtype)
        df_numerical = df.drop(columns=['patch_id', 'patch_id_s1'])
        df_description = df_numerical.describe()
        print(df_description)
        print("Timestamp S1: \n"+str(df['timestamp_s1'].describe()))
        print("Timestamp S2: \n"+str(df['timestamp_s2'].describe()))
        print("stop")