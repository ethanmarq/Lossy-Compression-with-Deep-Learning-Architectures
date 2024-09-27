#!/usr/bin/env python

import json
import numpy as np
import sys
import os
import numpy as np

import os
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

compressor = "jpeg"
#0 is base dataset no compression, 1 is 1E-1, 2 is 1E-2, etc up to 1E-7
csv_path = f"/scratch/aniemcz/cavResnetUnet/{compressor}/UNET_train_{compressor}.csv"
save_path = f"results/{compressor}/"

os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
df = pd.read_csv(csv_path)
#df['error_bound'] = df['error_bound'].astype(str)

new_values = ['Q100', 'Q80', 'Q60', 'Q40', 'Q20', 'Q0']
#new_values = [7, 6, 5, 4, 3, 2, 1]
'''
repeat_count = 4  # Number of times each value should appear (4 classes so each bound has 4 rows with each row per class)

index = 0
for new_value in new_values:
    for _ in range(repeat_count):
        df.at[index, 'error_bound'] = new_value
        index += 1
'''
print(df.dtypes)
print(df)

sns.set(rc={"figure.figsize":(24, 10)})
sns.set_context("talk", font_scale=1)  # Increases overall font size, change font_scale to adjust size

# Set custom font sizes using rcParams
plt.rcParams.update({
    'axes.titlesize': 20*1.9,   # Title font size
    'axes.labelsize': 18*1.9,   # X and Y label font size
    'xtick.labelsize': 16*1.9,  # X-tick label font size
    'ytick.labelsize': 16*1.9,  # Y-tick label font size
    'legend.fontsize': 14*1.9   # Legend font size
})
line_plot = sns.lineplot(data=df, x="error_bound", y="miou", hue = "Compressor")

xlab = "jpeg Quality"
ylab = "miou"
title = f"{compressor} UNET iou scores"
sns.move_legend(line_plot, "upper left", bbox_to_anchor=(1, 1))


plt.xlabel('Error Bound')
plt.ylabel('miou')

line_plot.set(title=f'Resnet-UNet mIoU on {compressor} compressor')

fig = line_plot.get_figure()

fig.savefig(
    os.path.join(save_path, f'{compressor}_unet_graph_cavs.svg')       
)



