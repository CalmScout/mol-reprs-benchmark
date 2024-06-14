"""
Tools for benchmarking and analyzing results.
"""
import polars as pl


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