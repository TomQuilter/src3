import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.parse_paper import parse_paper
from utils.df_utils.combine_df import combine_df
from utils.df_utils.threshold_df import threshold_df
from utils.df_utils.binarise_df import binarise_df

exam_df1, meta_df1 = parse_paper('new1')
exam_df2, meta_df2 = parse_paper('new2')
exam_df3, meta_df3 = parse_paper('new3')

combined_exam_df = combine_df([exam_df1, exam_df2, exam_df3])
combined_meta_df = combine_df([meta_df1, meta_df2, meta_df3])

max_scores_df = combined_meta_df.loc['Max'].astype(float)
thres_df = threshold_df(combined_exam_df, max_scores_df)
bin_df = binarise_df(thres_df, max_scores_df)

data_df = bin_df
question_names = data_df.columns

data_ts = torch.tensor(data_df.values)
is_nan = ~torch.isnan(data_ts)
total = torch.sum(is_nan, 1)
print(total)

correct = torch.nansum(data_ts, 1)
print(correct)
frac_correct = correct/total
print(frac_correct)

plt.hist(frac_correct.numpy(), bins=11)
plt.ylabel('Number of students')
plt.xlabel('Percentage correctness for each student')
plt.show()
