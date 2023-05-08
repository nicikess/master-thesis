import pandas as pd


def check_if_present(patch_id, df_train, df_validation, df_test):
    is_present_train = patch_id in df_train["patch_id"].values
    is_present_validation = patch_id in df_validation["patch_id"].values
    is_present_test = patch_id in df_test["patch_id"].values

    if not is_present_train and not is_present_validation and not is_present_test:
        return True
    else:
        return False


def make_split():
    # Path to store data files
    data_path = "/netscratch2/nkesseli/master-thesis-benge/src/master_thesis_benge/scripts/data-split/data-split-folder-2/"

    # Load ben-ge
    df_large = pd.read_csv(
        "/ds2/remote_sensing/ben-ge/ben-ge/esaworldcover/npy/ben-ge_esaworldcover.csv"
    )

    # Load ben-ge-s
    df_small_train = pd.read_csv(
        "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-train.csv"
    )
    df_small_validation = pd.read_csv(
        "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-validation.csv"
    )
    df_small_test = pd.read_csv(
        "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-test.csv"
    )

    # Initialize data splits
    df_train = df_small_train
    df_validation = df_small_validation
    df_test = df_small_test

    # Define iterator
    # iterator = df_large.iterrows()

    dict_size = {
        "20": 116464,
        "40": 232928,
        "60": 349392,
        "80": 465856
        # "100": 582320,
    }

    for key, value in dict_size.items():
        # Define sizes
        len_train = int(value * 0.8)
        len_validation_and_test = int(value * 0.1)

        for index, row in df_large.iterrows():
            if index % 10000 == 0:
                print("Index: " + str(index))
            patch_id = row.loc["patch_id"]
            not_present = check_if_present(patch_id, df_train, df_validation, df_test)
            if not_present and len(df_train) < len_train:
                df_train = pd.concat([df_train, row.to_frame().T], ignore_index=True)
                continue
            if not_present and len(df_validation) < len_validation_and_test:
                df_validation = pd.concat(
                    [df_validation, row.to_frame().T], ignore_index=True
                )
                continue
            if not_present and len(df_test) < len_validation_and_test:
                df_test = pd.concat([df_test, row.to_frame().T], ignore_index=True)
                continue

        df_train.to_csv(data_path + "ben-ge-train" + key + ".csv", index=False)
        df_validation.to_csv(
            data_path + "ben-ge-validation" + key + ".csv", index=False
        )
        df_test.to_csv(data_path + "ben-ge-test" + key + ".csv", index=False)


def check_split_for_unique_patch_ids():
    # Load the three CSV files into separate DataFrames
    df1 = pd.read_csv("data-split-folder/ben-ge-train.csv")
    df2 = pd.read_csv("data-split-folder/ben-ge-validation.csv")
    df3 = pd.read_csv("data-split-folder/ben-ge-test.csv")

    print("Length train: " + str(len(df1)))
    print("Length train: " + str(len(df2)))
    print("Length train: " + str(len(df3)))

    # Select the column to check for uniqueness
    column_name = "patch_id"

    # Concatenate the columns from all three DataFrames into a single DataFrame
    concatenated_df = pd.concat([df1[column_name], df2[column_name], df3[column_name]])

    # Count the number of unique values in the concatenated DataFrame
    num_unique = concatenated_df.nunique()

    # Compare the number of unique values to the total number of rows in the concatenated DataFrame
    if num_unique == concatenated_df.shape[0]:
        print("All values in the column are unique across all three CSV files.")
    else:
        print("There are duplicate values in the column across the three CSV files.")


def is_subset(df1, df2):
    """
    Checks if all the values of a column in df1 are included in df2.

    Args:
        df1 (pandas.DataFrame): The first DataFrame.
        df2 (pandas.DataFrame): The second DataFrame.
        col_name (str): The name of the column to check.

    Returns:
        bool: True if all the values of the column in df1 are included in df2, False otherwise.
    """
    col_name = "patch_id"

    values1 = set(df1[col_name].values)
    values2 = set(df2[col_name].values)
    return values1.issubset(values2)


def check_column_non_inclusion(df1, df2):
    """
    Checks if all the values of a column in one dataframe are not included in another one.

    Args:
    df1 (pandas.DataFrame): The first DataFrame.
    df2 (pandas.DataFrame): The second DataFrame.
    column (str): The name of the column to check.

    Returns:
    bool: True if none of the values in the specified column of df1 are included in df2, False otherwise.
    """

    column = "patch_id"

    values1 = set(df1[column])
    values2 = set(df2[column])
    return not any(value in values2 for value in values1)


if __name__ == "__main__":

    #make_split()

    # Check that all values from ben-ge-s are in cluded in the respective train,validation,test splits
    df_small_train = pd.read_csv(
        "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-train.csv"
    )
    df_small_validation = pd.read_csv(
        "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-validation.csv"
    )
    df_small_test = pd.read_csv(
        "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-test.csv"
    )

    # Load split data for train
    print('Train subsets check')
    df_train20 = pd.read_csv("data-split-folder-2/ben-ge-train20.csv")
    df_train40 = pd.read_csv("data-split-folder-2/ben-ge-train40.csv")
    df_train60 = pd.read_csv("data-split-folder-2/ben-ge-train60.csv")
    df_train80 = pd.read_csv("data-split-folder-2/ben-ge-train80.csv")
    df_train = pd.read_csv("data-split-folder/ben-ge-train.csv")
    print(is_subset(df_small_train, df_train20))
    print(is_subset(df_small_train, df_train40))
    print(is_subset(df_small_train, df_train60))
    print(is_subset(df_small_train, df_train80))
    print(is_subset(df_small_train, df_train))
    print('length 20: '+str(len(df_train20)))
    print('length 40: '+str(len(df_train40)))
    print('length 60: '+str(len(df_train60)))
    print('length 80: '+str(len(df_train80)))
    print('length 100: '+str(len(df_train)))
    print('')

    print('Validation subsets check')
    df_validation20 = pd.read_csv("data-split-folder-2/ben-ge-validation20.csv")
    df_validation40 = pd.read_csv("data-split-folder-2/ben-ge-validation40.csv")
    df_validation60 = pd.read_csv("data-split-folder-2/ben-ge-validation60.csv")
    df_validation80 = pd.read_csv("data-split-folder-2/ben-ge-validation80.csv")
    df_validation = pd.read_csv("data-split-folder/ben-ge-validation.csv")
    print(is_subset(df_small_validation, df_validation20))
    print(is_subset(df_small_validation, df_validation40))
    print(is_subset(df_small_validation, df_validation60))
    print(is_subset(df_small_validation, df_validation80))
    print(is_subset(df_small_validation, df_validation))
    print('length 20: '+str(len(df_validation20)))
    print('length 40: '+str(len(df_validation40)))
    print('length 60: '+str(len(df_validation60)))
    print('length 80: '+str(len(df_validation80)))
    print('length 100: '+str(len(df_validation)))
    print('')

    print('Test subsets check')
    df_test20 = pd.read_csv("data-split-folder-2/ben-ge-test20.csv")
    df_test40 = pd.read_csv("data-split-folder-2/ben-ge-test40.csv")
    df_test60 = pd.read_csv("data-split-folder-2/ben-ge-test60.csv")
    df_test80 = pd.read_csv("data-split-folder-2/ben-ge-test80.csv")
    df_test = pd.read_csv("data-split-folder/ben-ge-test.csv")
    print(is_subset(df_small_test, df_test20))
    print(is_subset(df_small_test, df_test40))
    print(is_subset(df_small_test, df_test60))
    print(is_subset(df_small_test, df_test80))
    print(is_subset(df_small_test, df_test))
    print('length 20: '+str(len(df_test20)))
    print('length 40: '+str(len(df_test40)))
    print('length 60: '+str(len(df_test60)))
    print('length 80: '+str(len(df_test80)))
    print('length 100: '+str(len(df_test)))
    print('')

    print('20 size exlusion')
    df_validation20 = pd.read_csv("data-split-folder-2/ben-ge-validation20.csv")
    df_test20 = pd.read_csv("data-split-folder-2/ben-ge-test20.csv")
    print(check_column_non_inclusion(df_train20, df_validation20))
    print(check_column_non_inclusion(df_train20, df_test20))
    print(check_column_non_inclusion(df_test20, df_validation20))
    print('')

    print('40 size exlusion')
    df_validation40 = pd.read_csv("data-split-folder-2/ben-ge-validation40.csv")
    df_test40 = pd.read_csv("data-split-folder-2/ben-ge-test40.csv")
    print(check_column_non_inclusion(df_train40, df_validation40))
    print(check_column_non_inclusion(df_train40, df_test40))
    print(check_column_non_inclusion(df_test40, df_validation40))
    print('')
    
    print('60 size exlusion')
    df_validation60 = pd.read_csv("data-split-folder-2/ben-ge-validation60.csv")
    df_test60 = pd.read_csv("data-split-folder-2/ben-ge-test60.csv")
    print(check_column_non_inclusion(df_train60, df_validation60))
    print(check_column_non_inclusion(df_train60, df_test60))
    print(check_column_non_inclusion(df_test60, df_validation60))
    print('')

    print('80 size exlusion')
    df_validation80 = pd.read_csv("data-split-folder-2/ben-ge-validation80.csv")
    df_test80 = pd.read_csv("data-split-folder-2/ben-ge-test80.csv")
    print(check_column_non_inclusion(df_train80, df_validation80))
    print(check_column_non_inclusion(df_train80, df_test80))
    print(check_column_non_inclusion(df_test80, df_validation80))
    print('')