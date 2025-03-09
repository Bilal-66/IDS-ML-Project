import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_nsl_kdd(train_path="data/KDDTrain+.txt", test_path="data/KDDTest+.txt"):
    """
    Read KDDTrain+.txt and KDDTest+.txt files (no header).
    Returns two pandas DataFrames: df_train and df_test.
    """
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)
    return df_train, df_test

def rename_columns(df):
    """
    The NSL-KDD dataset (KDDTrain+, KDDTest+) has 43 columns (index 0..42).
    - col41 = label (e.g., 'normal', 'neptune', 'mscan', etc.)
    - col42 = difficulty (e.g., 21, 15, etc.)

    Rename:
      col41 => 'label'
      col42 => 'difficulty'
    The other columns remain col0, col1, ..., col40.
    """
    num_cols = df.shape[1]  # Should be 43
    col_names = [f"col{i}" for i in range(num_cols)]
    df.columns = col_names

    # Rename column 41 to "label" and column 42 to "difficulty"
    df.rename(columns={"col41": "label", "col42": "difficulty"}, inplace=True)
    return df

def encode_categorical(df):
    """
    Encode categorical features ('protocol_type', 'service', 'flag') into numerical values.
    """
    cat_cols = ["col1", "col2", "col3"]  # Columns containing categorical data
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Transform values into numerical format

    return df

def basic_preprocessing(df):
    """
    1) Rename columns (to identify 'label' and 'difficulty')
    2) Encode col1, col2, col3
    """
    df = rename_columns(df)
    df = encode_categorical(df)
    return df

def split_dataset(df, test_size=0.2):
    """
    Split the DataFrame into features (X) and label (y).
    - 'label' and 'difficulty' are removed from the features,
      as 'difficulty' is not needed for ML,
      and 'label' is the target to predict.
    """
    X = df.drop(columns=["label", "difficulty"])
    y = df["label"]

    return train_test_split(X, y, test_size=test_size, random_state=42)
