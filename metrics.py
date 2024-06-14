"""
Module defines the class for metrics calculations.
"""
import polars as pl
from typing import Callable
from sklearn.metrics import matthews_corrcoef

class ClassificationMetrics:
    """
    Class for classification metrics calculations.
    """
    def __init__(self, model:Callable, df:pl.DataFrame, target_id:str):
        """
        Parameters
        ----------
        model: Callable
            Model to be used for predictions.
        df: pl.DataFrame
            Polars dataframe with features `f_<id>`, train/validation/test
            split in `split` in `split` column, and labels `class_label`.
        target_id: str
            Target id to be used for metrics calculations.
        """
        assert "class_label" in df.columns and "split" in df.columns
        self.target_id = target_id

        df_train = df.filter(pl.col("split") == "train")
        df_val = df.filter(pl.col("split") == "validation")
        df_test = df.filter(pl.col("split") == "test")

        feature_cols = [f for f in df.columns if f.startswith("f_")]

        X_train, self.y_train = df_train[feature_cols], df_train["class_label"]
        X_val, self.y_val = df_val[feature_cols], df_val["class_label"]
        X_test, self.y_test = df_test[feature_cols], df_test["class_label"]

        self.y_train_pred = model.predict(X_train)
        self.y_val_pred = model.predict(X_val)
        self.y_test_pred = model.predict(X_test)

        # calculate matthews correlation coefficient
        self.mcc_train = self.mcc(self.y_train, self.y_train_pred)
        self.mcc_val = self.mcc(self.y_val, self.y_val_pred)
        self.mcc_test = self.mcc(self.y_test, self.y_test_pred)

    def mcc(self, y, y_pred):
        """
        Calculates the Matthews correlation coefficient.
        """
        return matthews_corrcoef(y, y_pred)
    
    def __repr__(self):
        return f"Target: {self.target_id}, train MCC: {self.mcc_train:.4f}, val MCC: {self.mcc_val:.4f}, test MCC: {self.mcc_test:.4f}.\n"