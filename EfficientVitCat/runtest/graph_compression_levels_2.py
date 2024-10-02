#!/usr/bin/env python

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Uncomment this if you want to pass CSV file paths as command-line arguments
# if len(sys.argv) < 4:
#    raise Exception("Pass paths of 3 CSV files as arguments")

# Define the mapping between error levels
# q_to_numeric_mapping = {
#     'Q0': '1E-7', 'Q10': '1E-6', 'Q20': '1E-5',
#     'Q30': '1E-4', 'Q40': '1E-3', 'Q50': '1E-2',
#     'Q60': '1E-1'
# }

# Data from CSV files
dataframes = []

# Simulating reading from CSV paths (replace with actual file paths or use command-line arguments)
for csv_path in sys.argv[1:4]:
    df = pd.read_csv(csv_path, dtype={'error_bound': str})
    
    # Ensure 'error_bound' is treated as a string
    df['error_bound'] = df['error_bound'].astype(str)
    
    # Create a combined error bound column (if needed)
    # df['combined_error_bound'] = df['error_bound'].apply(
    #    lambda x: f"{x} ({q_to_numeric_mapping[x]})" if x in q_to_numeric_mapping else x
    # )
    
    dataframes.append(df)

# Concatenate dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Debug prints to check data
print(combined_df.dtypes)
print(combined_df)

# Set up the plot using seaborn
sns.set(rc={"figure.figsize": (22, 10)})  # Adjusting figure size similar to the first script
sns.set_context("talk", font_scale=1)  # Set font scale for better readability

# You can set a specific style like white grid for better clarity
# sns.set_style("whitegrid")

# Update the plot's default parameters for better visualization
plt.rcParams.update({
    'xtick.labelsize': 25,   # X-tick label font size
    'ytick.labelsize': 25,   # Y-tick label font size
    'legend.fontsize': 20    # Legend font size
})

# Convert 'combined_error_bound' to categorical data type for proper ordering (if applicable)
# combined_df['combined_error_bound'] = pd.Categorical(
#    combined_df['combined_error_bound'], 
#    categories=sorted(combined_df['combined_error_bound'].unique(), key=lambda x: (x.split()[0], x)),
#    ordered=True
# )

# Plot mIoU for each compressor across the error bounds
line_plot = sns.lineplot(data=combined_df, x="error_bound", y="miou", hue="Compressor", marker="o")

# Customize the plot
plt.xlabel('Compression Level (Error Bound)', fontsize=30, labelpad=10)
plt.ylabel('mIoU', fontsize=30, labelpad=10)
plt.title('EfficientVit mIoU on JPEG compressor', fontsize=35, pad=20)

# Rotate x-axis labels to avoid overlap
plt.xticks(rotation=45, ha='right')

# Move Legend (adjust the position and bbox)
sns.move_legend(line_plot, "upper left", bbox_to_anchor=(1, 1))

# Apply tight layout to prevent clipping of labels and legends
plt.tight_layout()

# Save the plot as an SVG file with tight bounding box to prevent clipping
save_path = '/home/marque6/MLBD/LossyUGVPaper/EfficientVitCat/runtest/'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(os.path.join(save_path, 'two_miou_comparison_jpeg.svg'), format='svg', bbox_inches='tight')

# Show the plot
plt.show()
