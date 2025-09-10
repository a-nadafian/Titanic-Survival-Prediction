import os
import pandas as pd

def load_data(data_dir):
    """Loads the Titanic training and test datasets.

    Args:
        data_dir (str): The directory where the raw data is stored.

    Returns:
        tuple: A tuple containing two pandas DataFrames: (train_df, test_df).
    """
    train_path = os.path.join(data_dir, "raw", "train.csv")
    test_path = os.path.join(data_dir, "raw", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def combine_data(train_df, test_df):
    """Combines the training and test dataframes for consistent preprocessing.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        test_df (pd.DataFrame): The test dataframe.

    Returns:
        pd.DataFrame: The combined dataframe.
    """
    df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    return df

def impute_embarked(df):
    """Imputes missing 'Embarked' values with the mode.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with 'Embarked' imputed.
    """
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    return df

def impute_fare(df):
    """Imputes missing 'Fare' values with the median fare of the passenger's class.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with 'Fare' imputed.
    """
    if df['Fare'].isnull().sum() > 0:
        pclass_for_missing_fare = df[df['Fare'].isnull()]['Pclass'].values[0]
        median_fare_for_pclass = df[df['Pclass'] == pclass_for_missing_fare]['Fare'].median()
        df['Fare'] = df['Fare'].fillna(median_fare_for_pclass)
    return df

def preprocess(data_dir):
    """Loads, combines, and preprocesses the Titanic data.

    Args:
        data_dir (str): The directory where the raw data is stored.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    train_df, test_df = load_data(data_dir)
    df = combine_data(train_df, test_df)
    df = impute_embarked(df)
    df = impute_fare(df)
    return df
