import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from sklearn.model_selection import train_test_split


def print_fractions():
    # plt.pie(overall_fractions.values, labels=overall_fractions.index)
    plt.show()


def sklearn_split(df, split_size):
    # Regular train_test split
    X_train, X_rem = train_test_split(df, train_size=split_size)
    overall_fractions = X_train.loc[:, "tree_cover":].sum(axis=0) / len(X_train)
    return X_train, X_rem


def get_n_weighted_samples(df, number_of_samples):
    df2 = pd.DataFrame()
    used_ids = []
    for c in df.columns:
        if c in [
            "filename",
            "patch_id",
            "snow_and_ice",
            "mangroves",
            "moss_and_lichen",
        ]:
            continue
        # extract 5000 tiles with highest coverage of this class
        # extract_ids = df.sort_values(c, ascending=False).index[:5000]

        # sample 1000 random tiles weighted by class-related coverage
        extract = df.sample(n=number_of_samples, replace=False, weights=df.loc[:, c])
        print(extract.to_string())
        df2 = pd.concat([df2, extract])
        df.drop(extract.index, inplace=True)
    return df2


def check_indices(df_train, df_validation, df_test):
    filename = df_train.iloc[0, "patch_id"]
    assert filename not in df_validation["patch_id"]
    assert filename not in df_validation["patch_id"]


def create_split_ben_ge_s():
    # BEN-GE-S
    # Set path and file name
    data_index_path_s = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/"
    data_index_path_to_store = "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/"
    file_name = "ben-ge-s_esaworldcover.csv"
    # Load .csv
    df = pd.read_csv(data_index_path_s + file_name)
    len_df = len(df)
    df_train, df_remain = sklearn_split(df, 0.8)
    assert len(df_train) == len_df * 0.8
    df_validation, df_test = sklearn_split(df_remain, 0.5)
    assert len(df_validation) == len_df * 0.1
    assert len(df_test) == len_df * 0.1
    # df_train = df_train.reset_index(drop=True)
    # df_validation = df_validation.reset_index(drop=True)
    # df_test = df_test.reset_index(drop=True)
    # check_indices(df_train, df_validation, df_test)
    df_train.to_csv(data_index_path_to_store + "ben-ge-s-train.csv", index=False)
    df_validation.to_csv(
        data_index_path_to_store + "ben-ge-s-validation.csv", index=False
    )
    df_test.to_csv(data_index_path_to_store + "ben-ge-s-test.csv", index=False)


def create_split_ben_ge_m():
    # BEN-GE-M
    data_index_path_m = "/ds2/remote_sensing/ben-ge/ben-ge-m/s2_npy/"
    data_index_path_to_store = "/ds2/remote_sensing/ben-ge/ben-ge-m/data-index/"
    file_name = "ben-ge-m_esaworldcover.csv"
    # Load .csv
    df = pd.read_csv(data_index_path_m + file_name)
    len_df = len(df)
    df_train, df_remain = sklearn_split(df, 0.8)
    assert len(df_train) == len_df * 0.8
    df_validation, df_test = sklearn_split(df_remain, 0.5)
    assert len(df_validation) == len_df * 0.1
    assert len(df_test) == len_df * 0.1
    # df_train = df_train.reset_index(drop=True)
    # df_validation = df_validation.reset_index(drop=True)
    # df_test = df_test.reset_index(drop=True)
    # check_indices(df_train, df_validation, df_test)
    df_train.to_csv(data_index_path_to_store + "ben-ge-m-train.csv", index=False)
    df_validation.to_csv(
        data_index_path_to_store + "ben-ge-m-validation.csv", index=False
    )
    df_test.to_csv(data_index_path_to_store + "ben-ge-m-test.csv", index=False)


def create_split_ben_ge_l():
    # BEN-GE
    data_index_path_l = "/ds2/remote_sensing/ben-ge/ben-ge/esaworldcover/npy/"
    data_index_path_to_store = "/ds2/remote_sensing/ben-ge/ben-ge/data-index/"
    file_name = "ben-ge_esaworldcover.csv"
    # Load .csv
    df = pd.read_csv(data_index_path_l + file_name)
    # Remove last 8 rows to make it divisible by 8
    df = df.iloc[0:582320]
    df_train, df_remain = sklearn_split(df, 0.8)
    len_df = len(df)
    assert len(df_train) == len_df * 0.8
    df_validation, df_test = sklearn_split(df_remain, 0.5)
    assert len(df_validation) == len_df * 0.1
    assert len(df_test) == len_df * 0.1
    # df_train = df_train.reset_index(drop=True)
    # df_validation = df_validation.reset_index(drop=True)
    # df_test = df_test.reset_index(drop=True)
    # check_indices(df_train, df_validation, df_test)
    df_train.to_csv(data_index_path_to_store + "ben-ge-train.csv", index=False)
    df_validation.to_csv(
        data_index_path_to_store + "ben-ge-validation.csv", index=False
    )
    df_test.to_csv(data_index_path_to_store + "ben-ge-test.csv", index=False)


if __name__ == "__main__":
    create_split_ben_ge_s()
    create_split_ben_ge_m()
    create_split_ben_ge_l()
