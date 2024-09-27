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

'''
if len(sys.argv) > 1:
    compressor = sys.argv[1]
    #0 is base dataset no compression, 1 is 1E-1, 2 is 1E-2, etc up to 1E-7
    csv_path = f"/scratch/aniemcz/cavResnetUnet/{compressor}/UNET_train_{compressor}.csv"
    save_path = f"results/{compressor}/"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
else:
    raise Exception("Need to pass the compressor name as arguments")
'''

sz3_csv_path = f"/scratch/aniemcz/cavResnetUnet/sz3/UNET_train_sz3.csv"
zfp_csv_path = f"/scratch/aniemcz/cavResnetUnet/zfp/UNET_train_zfp.csv"
save_path = f"results/"
compressor = "sz3 and zfp"

sz3_df = pd.read_csv(sz3_csv_path, dtype={'error_bound': 'string'})
zfp_df = pd.read_csv(zfp_csv_path, dtype={'error_bound': 'string'})

df = pd.concat([sz3_df, zfp_df], ignore_index=True, )

new_values = ['1E-7', '1E-6', '1E-5', '1E-4', '1E-3', '1E-2', '1E-1']

print(df.dtypes)
print(df)

sns.set(rc={"figure.figsize":(22, 10)})
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

plt.gca().invert_xaxis() #flip x axis

xlab = "error bound"
ylab = "miou"
title = f"{compressor} UNET iou scores"
sns.move_legend(line_plot, "upper left", bbox_to_anchor=(1, 1))


plt.xlabel('Error Bound')
plt.ylabel('iou')

line_plot.set(title=f'Resnet-UNet mIoU on {compressor} compressors')

fig = line_plot.get_figure()

fig.savefig(
    os.path.join(save_path, f'{compressor}_unet_graph_cavs.svg')       
)



