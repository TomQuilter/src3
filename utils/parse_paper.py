import pandas as pd
from typing import List, Tuple
#data/9to1_2017_GCSE_1H_and_2H_and_3H Linked Pinpoint Data_Cleaned.csv
 
def parse_paper(paper: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if paper == 'old':
        exam_df, meta_df = parse_csv(
            "data/9to1_2017_GCSE_1H.csv",
            data_row_start=23,
            meta_rows=5,
            paper_columns=['Name'] + [f'q{i}' for i in range(1, 25)])
 
    elif paper == 'new1': 
        exam_df, meta_df = parse_csv(  
            "data/LargeDataSet13_HigherExamsWithClassAndSchoolsAvailableSmall.csv",   
            data_row_start=6, 
            meta_rows=5, 
            paper_columns=['Name'] + [f'q{i}' for i in range(1, 25)])

    elif paper == 'new2':
        exam_df, meta_df = parse_csv(
            "data/LargeDataSet13_HigherExamsWithClassAndSchoolsAvailable.csv",
            data_row_start=6,
            meta_rows=5,
            paper_columns=['Name.1'] + [f'q{i}.1' for i in range(1, 24)])

    elif paper == 'new3':
        exam_df, meta_df = parse_csv(
            "data/LargeDataSet13_HigherExamsWithClassAndSchoolsAvailableSMALL.csv",
            data_row_start=6,
            meta_rows=5,
            paper_columns=['Name.2'] + [f'q{i}.2' for i in range(1, 24)])

    return exam_df, meta_df


def parse_csv(csv_name: str, data_row_start: int, meta_rows: int, paper_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    returned exam_df has original row index retained which represents unique student id
    e.g. row index 6 -> student 6
    '''
    raw_data = pd.read_csv(csv_name, low_memory=False)
    data = raw_data[paper_columns]

    meta_df = data.head(meta_rows)
    meta_df = meta_df.set_index(paper_columns[0]) # set 'Name, Max, Topics' as (row) index

    exam_df = data[data_row_start:]
    exam_df = exam_df.dropna(axis=0) # drop rows with nan values, (row) index retained
    exam_df = exam_df.astype(float)
    exam_df = exam_df[meta_df.columns] # meta_df.columns: ['q1', 'q2', ..., 'q24']

    return exam_df.copy(), meta_df.copy()
