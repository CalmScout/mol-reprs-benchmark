"""
Tools for benchmarking and analyzing results.
"""
import polars as pl
from metrics import ClassificationMetrics


def expand_array_column(df, col_name, array_length):
    # Generate column names
    new_columns = [f"f_{i+1:04d}" for i in range(array_length)]
    
    # Initialize an empty dictionary to store new columns
    expanded_columns = {new_col: [] for new_col in new_columns}
    
    # Iterate through the rows of the DataFrame
    for array in df[col_name]:
        for i, value in enumerate(array):
            expanded_columns[f"f_{i+1:04d}"].append(value)
    
    # Create a new DataFrame with the expanded columns
    expanded_df = pl.DataFrame(expanded_columns)
    
    # Concatenate the original DataFrame (without the array column) with the expanded DataFrame
    result_df = df.drop([col_name]).hstack(expanded_df)
    
    return result_df


def train_classifier(model, df_target, target_id):
    """
    Parameters
    ----------
    model: Callable
        Model to be used for predictions.
    df_target: pl.DataFrame
        Polars dataframe with features `f_<id>`, train/validation/test
        split in `split` in `split` column, and labels `class_label`
        and features `f_<id>, id in `range(1, 2049)`.
    target_id: str
        Target id to be used for metrics calculations.
    """
    assert hasattr(model, "fit")
    assert "split" in df_target.columns

    df_train = df_target.filter(pl.col("split") == "train")
    # df_val = df_target.filter(pl.col("split") == "validation")
    # df_test = df_target.filter(pl.col("split") == "test")

    feature_cols = [f"f_{i+1:04d}" for i in range(2048)]
    target_col = "class_label"

    X_train, y_train = df_train[feature_cols], df_train[target_col]
    
    # Train the model
    model.fit(X_train, y_train)
    
    metrics = ClassificationMetrics(model, df_target, target_id)
    return metrics
