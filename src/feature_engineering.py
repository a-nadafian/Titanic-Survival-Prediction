import re
import pandas as pd

def add_family_features(df):
    """Adds family size and family size group features.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with new family features.
    """
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilySize_Group'] = 'Medium'
    df.loc[df['FamilySize'] == 1, 'FamilySize_Group'] = 'Alone'
    df.loc[df['FamilySize'] >= 5, 'FamilySize_Group'] = 'Large'
    return df

def add_cabin_features(df):
    """Adds features based on the 'Cabin' column.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with new cabin features.
    """
    df['CabinAssigned'] = df['Cabin'].notna().astype(int)
    df['Deck'] = df['Cabin'].fillna('U').apply(lambda x: x[0])
    return df

def extract_title(df):
    """Extracts and consolidates titles from the 'Name' column.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with a new 'Title' column.
    """
    def get_title(name):
        match = re.search(r' ([A-Za-z]+)\\.', name)
        if match:
            return match.group(1)
        return 'Unknown'

    df['Title'] = df['Name'].apply(get_title)
    title_mapping = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Capt': 'Official',
        'Col': 'Official', 'Major': 'Official', 'Rev': 'Official',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare',
        'Sir': 'Rare', 'Lady': 'Rare', 'Countess': 'Rare'
    }
    df['Title'] = df['Title'].replace(title_mapping)
    return df

def impute_age(df):
    """Imputes missing 'Age' values based on title-specific medians.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with 'Age' imputed.
    """
    median_ages = df.groupby('Title')['Age'].median()
    for title in median_ages.index:
        df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = median_ages[title]
    return df

def add_ticket_features(df):
    """Adds features based on the 'Ticket' column.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with new ticket features.
    """
    df['Ticket_Prefix'] = df['Ticket'].apply(lambda x: x.split()[0] if not x.split()[0].isdigit() else 'NUM')
    df['Ticket_Prefix'] = df['Ticket_Prefix'].str.replace(r'[\\./]', '', regex=True)
    df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')
    return df

def add_age_bins(df):
    """Adds age bins to the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with a new 'AgeBin' column.
    """
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 120], labels=['Child', 'Teenage', 'Adult', 'Elder'])
    return df

def add_fare_bins(df):
    """Adds fare bins to the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with a new 'FareBin' column.
    """
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=['Very_Low', 'Low', 'High', 'Very_High'])
    return df

def add_interaction_features(df):
    """Adds interaction features.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with new interaction features.
    """
    df['Age_Class'] = df['Age'] * df['Pclass']
    df['Fare_per_Person'] = df['Fare'] / df['FamilySize']
    return df

def encode_categoricals(df):
    """One-hot encodes all categorical features.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with categorical features encoded.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def drop_original_columns(df):
    """Drops original columns that are no longer needed.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with original columns dropped.
    """
    cols_to_drop = ['Name', 'Ticket', 'SibSp', 'Parch', 'Age', 'Fare', 'Cabin']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

def engineer_features(df):
    """Engineers all features for the Titanic dataset.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The dataframe with all features engineered.
    """
    df = extract_title(df)
    df = impute_age(df)
    df = add_family_features(df)
    df = add_cabin_features(df)
    df = add_ticket_features(df)
    df = add_age_bins(df)
    df = add_fare_bins(df)
    df = add_interaction_features(df)
    df = drop_original_columns(df)
    df = encode_categoricals(df)
    return df
