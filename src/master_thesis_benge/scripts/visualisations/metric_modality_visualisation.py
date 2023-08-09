import matplotlib.pyplot as plt

if __name__ == "__main__":
    # One modality
    one_modality = ["Sen-2", "Sen-1", "EWC", "DEM", "Weather", "Season"]
    one_modality_f1 = [89.76, 86.07, 88.49, 65.8, 65.8, 55.44]
    one_modality_acc = [99.32, 99.09, 99.24, 97.93, 96.72, 97.33]
    plt.plot(
        one_modality, one_modality_f1, marker="s", label="One modality", color="blue"
    )
    plt.plot(one_modality, one_modality_acc, marker="s", color="orange")

    # Two modalities
    two_modality = ["Sen-2+Sen-1", "Sen-2+EWC"]
    two_modality_f1 = [89.76, 86.07]
    two_modality_acc = [99.32, 99.09]
    plt.plot(
        two_modality, two_modality_f1, marker="v", label="Two modalities", color="blue"
    )
    plt.plot(two_modality, two_modality_acc, marker="v", color="orange")

    # Three modalities
    three_modality = ["Sen-2+Sen-1+Glo", "Sen-2+Sen-1+EWC"]
    three_modality_f1 = [89.76, 86.07]
    three_modality_acc = [99.32, 99.09]
    plt.plot(
        three_modality,
        three_modality_f1,
        marker="o",
        label="Three modalities",
        color="blue",
    )
    plt.plot(three_modality, three_modality_acc, marker="o", color="orange")

    # Add labels and title
    plt.xlabel("Modality")
    plt.ylabel("Score")
    plt.xticks(fontsize=6)
    plt.title("Performance of Model across Different Modalities")

    # Add a legend and position it outside the plot
    legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Change legend color to grey
    handles, labels = legend.legendHandles, legend.get_texts()
    handles[0].set_color("grey")
    handles[1].set_color("grey")
    handles[2].set_color("grey")

    # Save the plot
    path_to_save = "/Users/nicolaskesseli/NICOLAS_KESSELI/Programming/Lokal/master-thesis-doc/src/files/"
    file_name = "classification_climatezone_modality_comparison"
    format = ".png"
    plt.savefig(path_to_save + file_name + format, bbox_inches="tight")

    # Display the plot
    plt.tight_layout()
    plt.show()
