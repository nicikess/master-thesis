# Import for plot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Import for load data
import pandas as pd


def update_epoch_column(
    rgb, infra_red, all_channels, twenty_m_resolution, sixty_m_resolution, no_sixty_m_resolution
):
    num_epochs = 20
    epochs = list(range(1, num_epochs + 1))
    for data_frame in [
        rgb,
        infra_red,
        all_channels,
        twenty_m_resolution,
        sixty_m_resolution,
        no_sixty_m_resolution,
    ]:
        # Rename columns
        data_frame.rename(
            columns={
                "sentinel2 - Epoch val f1 score": "mean",
                "sentinel2 - Epoch val f1 score__MIN": "min",
                "sentinel2 - Epoch val f1 score__MAX": "max",
                "Step": "epoch",
            },
            inplace=True,
        )

        # Replace the "Step" column with the "Epoch" column (1 to 20)
        for i in range(len(data_frame)):
            data_frame.iloc[i, 0] = epochs[i]

    return rgb, infra_red, all_channels, twenty_m_resolution, sixty_m_resolution, no_sixty_m_resolution


def load_data():
    rgb = pd.read_csv("/Users/nicolaskesseli/Downloads/rgb.csv")
    infra_red = pd.read_csv("/Users/nicolaskesseli/Downloads/infra_red.csv")
    all_channels = pd.read_csv("/Users/nicolaskesseli/Downloads/all_channels.csv")
    twenty_m_resolution = pd.read_csv(
        "/Users/nicolaskesseli/Downloads/20m_resolution.csv"
    )
    sixty_m_resolution = pd.read_csv(
        "/Users/nicolaskesseli/Downloads/60m_resolution.csv"
    )
    no_sixty_m_resolution = pd.read_csv(
        "/Users/nicolaskesseli/Downloads/no_60m_resolution.csv"
    )

    for data_frame in [
        rgb,
        infra_red,
        all_channels,
        twenty_m_resolution,
        sixty_m_resolution,
        no_sixty_m_resolution,
    ]:
        print(data_frame.to_string())

    return update_epoch_column(
        rgb, infra_red, all_channels, twenty_m_resolution, sixty_m_resolution, no_sixty_m_resolution
    )


def make_plot(rgb, infra_red, all_channels, twenty_m_resolution, sixty_m_resolution, no_sixty_m_resolution):
    # Create a list of designs for each data frame
    designs = [
        {"label": "RGB", "color": "tab:blue", "marker": "o", "linestyle": "--"},
        {"label": "Infra Red", "color": "tab:red", "marker": "s", "linestyle": "--"},
        {"label": "All Channels", "color": "tab:green", "marker": "D", "linestyle": "-."},
        {"label": "20m Resolution", "color": "tab:orange", "marker": "^", "linestyle": "-."},
        {"label": "60m Resolution", "color": "tab:pink", "marker": "v", "linestyle": ":"},
        {"label": "No-60m Resolution", "color": "black", "marker": "X", "linestyle": ":"},
    ]

    # Plot each data frame with its corresponding design
    for i, data_frame in enumerate(
        [rgb, infra_red, all_channels, twenty_m_resolution, sixty_m_resolution, no_sixty_m_resolution]
    ):
        f1_score_err = data_frame.loc[:, "max"] - data_frame.loc[:, "min"]
        design = designs[i]

        plt.errorbar(
            data_frame.loc[:, "epoch"],
            data_frame.loc[:, "mean"],
            #yerr=f1_score_err,
            capsize=5,
            label=f"Sentinel2 - ({design['label']})",
            marker=design["marker"],
            markersize=8,
            linestyle=design["linestyle"],
            color=design["color"],
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("Performance on Validation Set", fontsize=14)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set the x-axis ticks explicitly to show all epochs
    epochs = list(range(1, len(rgb) + 1))
    plt.xticks(epochs, fontsize=10)

    plt.yticks(fontsize=10)

    # Add a background grid
    plt.grid(color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()  # Ensure all elements fit in the plot area
    plt.show()


if __name__ == "__main__":
    rgb, infra_red, all_channels, twenty_m_resolution, sixty_m_resolution, no_sixty_m_resolution = load_data()
    make_plot(rgb, infra_red, all_channels, twenty_m_resolution, sixty_m_resolution, no_sixty_m_resolution)
