import numpy as np
import pandas as pd


def reset_row_index(data_df: pd.DataFrame) -> pd.DataFrame:
    reset_data_df = data_df.copy()
    reset_data_df.reset_index(drop=True, inplace=True) # drop the old index, reset row index -> 0 to last_student_id,
    return reset_data_df

def reset_col_index(data_df: pd.DataFrame) -> pd.DataFrame:
    reset_data_df = data_df.copy()
    reset_data_df.columns = np.arange(len(data_df.columns)) # rename columns -> 0 to last_question_id
    return reset_data_df
