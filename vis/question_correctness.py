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
total = torch.sum(is_nan, 0)

correct = torch.nansum(data_ts, 0)
frac_correct = correct/total

paper1 = np.array(range(24)) # 24q
paper2 = np.array(range(24, 24+23)) # 23q
paper3 = np.array(range(24+23, 70)) # 23q

question_names = question_names.tolist()
xticks1 = question_names[:24]
xticks2 = question_names[24: 24+23]
xticks3 = question_names[24+23: 70]
question_names = ['' if i%2 == 1 else question_names[i] for i in range(len(question_names))]
print(question_names)

no_ones, no_twos, no_threes = 24, 23, 23
plt.bar(range(no_ones), frac_correct[:no_ones], label='Paper 1')
plt.bar(range(no_ones, no_ones+no_twos), frac_correct[no_ones:no_ones+no_twos], label='Paper 2')
plt.bar(range(no_ones+no_twos, no_ones+no_twos+no_threes), frac_correct[no_ones+no_twos:no_ones+no_twos+no_threes], label='Paper 3')
plt.xticks(range(70), question_names, rotation=60)

plt.legend()
plt.xlabel('question')
plt.ylabel('percentage correct')

plt.show()
