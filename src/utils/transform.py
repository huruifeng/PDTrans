import numpy as np
import pandas as pd

def max_min_normalization(df,feature_range=(0,1),on="col",inplace=False):
    """
    Normalize the data in the dataframe using the max-min normalization method.

    Parameters:
    df (DataFrame): The dataframe to normalize.
    feature_range (tuple): The range of the normalized data.
    on (str): The axis to normalize the data. It can be either "col" or "row".
    inplace (bool): Whether to modify the original dataframe or return a new one.

    Returns:
    DataFrame: The normalized dataframe.
    """
    if not inplace:
        df = df.copy()

    if on == "col":
        for col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * (feature_range[1] - feature_range[0]) + feature_range[0]
    elif on == "row":
        for i in range(len(df)):
            df.iloc[i] = (df.iloc[i] - df.iloc[i].min()) / (df.iloc[i].max() - df.iloc[i].min()) * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        raise ValueError("The 'on' parameter must be either 'col' or 'row'.")

    if not inplace:
        return df


def log_transformation(df,base=10,inplace=False):
    """
    Transform the data in the dataframe using the logarithm transformation method.

    Parameters:
    df (DataFrame): The dataframe to transform.
    base (int): The base of the logarithm.
    inplace (bool): Whether to modify the original dataframe or return a new one.

    Returns:
    DataFrame: The transformed dataframe.
    """
    if not inplace:
        df = df.copy()

    df = df.applymap(lambda x: np.log(x) / np.log(base))

    if not inplace:
        return df

def log1p_transformation(df,base=10,shift=1.0,inplace=False):
    """
    Transform the data in the dataframe using the log1p transformation method.

    Parameters:
    df (DataFrame): The dataframe to transform.
    base (int): The base of the logarithm.
    shift (float): The value to add before applying the logarithm.
    inplace (bool): Whether to modify the original dataframe or return a new one.

    Returns:
    DataFrame: The transformed dataframe.
    """
    if not inplace:
        df = df.copy()

    df = df.applymap(lambda x: np.log1p(x + shift) / np.log1p(base))

    if not inplace:
        return df



def z_score_normalization(df,on="col",inplace=False):
    """
    Normalize the data in the dataframe using the z-score normalization method.

    Parameters:
    df (DataFrame): The dataframe to normalize.
    on (str): The axis to normalize the data. It can be either "col" or "row".
    inplace (bool): Whether to modify the original dataframe or return a new one.

    Returns:
    DataFrame: The normalized dataframe.
    """
    if not inplace:
        df = df.copy()

    if on == "col":
        for col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    elif on == "row":
        for i in range(len(df)):
            df.iloc[i] = (df.iloc[i] - df.iloc[i].mean()) / df.iloc[i].std()
    else:
        raise ValueError("The 'on' parameter must be either 'col' or 'row'.")

    if not inplace:
        return df

def robust_normalization(df,on="col",inplace=False):
    """
    Normalize the data in the dataframe using the robust normalization method.

    Parameters:
    df (DataFrame): The dataframe to normalize.
    on (str): The axis to normalize the data. It can be either "col" or "row".
    inplace (bool): Whether to modify the original dataframe or return a new one.

    Returns:
    DataFrame: The normalized dataframe.
    """
    if not inplace:
        df = df.copy()

    if on == "col":
        for col in df.columns:
            df[col] = (df[col] - df[col].median()) / (df[col].quantile(0.75) - df[col].quantile(0.25))
    elif on == "row":
        for i in range(len(df)):
            df.iloc[i] = (df.iloc[i] - df.iloc[i].median()) / (df.iloc[i].quantile(0.75) - df.iloc[i].quantile(0.25))
    else:
        raise ValueError("The 'on' parameter must be either 'col' or 'row'.")

    if not inplace:
        return df

def binary_transformation(df,threshold=0.5,inplace=False):
    """
    Transform the data in the dataframe using the binary transformation method.

    Parameters:
    df (DataFrame): The dataframe to transform.
    threshold (float): The threshold value to apply the transformation.
    inplace (bool): Whether to modify the original dataframe or return a new one.

    Returns:
    DataFrame: The transformed dataframe.
    """
    if not inplace:
        df = df.copy()

    df = df.applymap(lambda x: 1 if x >= threshold else 0)

    if not inplace:
        return df


def rank_transformation(df,method="average",on="col",inplace=False):
    """
    Transform the data in the dataframe using the rank transformation method.

    Parameters:
    df (DataFrame): The dataframe to transform.
    method (str): The method to assign ranks. It can be "average", "min", "max", "dense", or "ordinal".
    on (str): The axis to transform the data. It can be either "col" or "row".
    inplace (bool): Whether to modify the original dataframe or return a new one.

    Returns:
    DataFrame: The transformed dataframe.
    """
    if not inplace:
        df = df.copy()

    if on == "col":
        df = df.rank(method=method)
    elif on == "row":
        df = df.T.rank(method=method).T
    else:
        raise ValueError("The 'on' parameter must be either 'col' or 'row'.")

    if not inplace:
        return df

def qrank_transformation(df,n_bins=100,on="col",inplace=False):
    """
    Transform the data in the dataframe using the quantile rank transformation method.

    Parameters:
    df (DataFrame): The dataframe to transform.
    n_bins (int): The number of bins to use.
    on (str): The axis to transform the data. It can be either "col" or "row".
    inplace (bool): Whether to modify the original dataframe or return a new one.

    Returns:
    DataFrame: The transformed dataframe.
    """
    if not inplace:
        df = df.copy()

    if on == "col":
        df = pd.qcut(df, q=n_bins, labels=False)
    elif on == "row":
        df = df.T.apply(lambda x: pd.qcut(x, q=n_bins, labels=False), axis=1).T
    else:
        raise ValueError("The 'on' parameter must be either 'col' or 'row'.")

    if not inplace:
        return df


def nrank_transformation(df,n_bins=100,on="col",inplace=False):
    """
    Transform the data in the dataframe using the bin rank transformation method.

    Parameters:
    df (DataFrame): The dataframe to transform.
    n_bins (int): The number of bins to use.
    on (str): The axis to transform the data. It can be either "col" or "row".
    inplace (bool): Whether to modify the original dataframe or return a new one.

    Returns:
    DataFrame: The transformed dataframe.
    """
    if not inplace:
        df = df.copy()

    if on == "col":
        df = pd.cut(df, bins=n_bins, labels=False)
    elif on == "row":
        df = df.T.apply(lambda x: pd.cut(x, bins=n_bins, labels=False), axis=1).T
    else:
        raise ValueError("The 'on' parameter must be either 'col' or 'row'.")

    if not inplace:
        return df
