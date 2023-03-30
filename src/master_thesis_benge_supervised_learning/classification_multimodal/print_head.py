import pandas as pd

if __name__ == "__main__":

    # path_both = "/ds2/remote_sensing/ben-ge/ben-ge-s/ben-ge-s_sentinel12_meta.csv"
    # path_both = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-1/s1_npy/ben-ge-s_esaworldcover.csv"
    esaworldcover_index = pd.read_csv(
        "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/ben-ge-s_esaworldcover.csv"
    )
    # path_both = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-1/s1_npy/ben-ge-s_esaworldcover.csv"
    # data = pd.read_csv(path_both)
    string = "S2A_MSIL2A_20171002T094031_64_46"
    # string = string.split("_")[3]+"_"+string.split("_")[4]
    # print(string)
    print(esaworldcover_index.head().to_string())
    label_vector = esaworldcover_index.loc[esaworldcover_index["patch_id"] == string]
    print(label_vector)
    label_vector = label_vector.drop(["filename", "patch_id"], axis=1)
    print(label_vector)
