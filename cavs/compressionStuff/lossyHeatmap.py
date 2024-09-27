import pandas as pd

lossy_df = pd.read_csv('output/cavs_mini_lossy_v2_Test.csv') #Id be skeptical about these results and also I am not using lossy anyway
jpeg_df = pd.read_csv('compression_metrics.csv')

compressors_dict = {}


import math
data = lossy_df[['compressor', 'CR', 'psnr', 'ssim', 'total_time', 'compressed_size_bytes', 'uncompressed_size_bytes', 'bound']]
jpeg_data = jpeg_df[['psnr', 'ssim', 'total_time', 'compressed_size_bytes', 'uncompressed_size_bytes', 'quality', 'split']]

compressors = ['SZ3', 'ZFP', 'JPEG', 'Uncompressed']
error_bounds = ['1E-7', '1E-6', '1E-5', '1E-4', '1E-3', '1E-2', '1E-1']
bounds = [math.exp(-7),math.exp(-6),math.exp(-5),math.exp(-4),math.exp(-3),math.exp(-2),math.exp(-1)]

quality_levels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

def isclose(x, ref_value):
    return math.isclose(x, ref_value)

for comp in compressors:
    if comp != 'Uncompressed' and comp != "JPEG":
        comp_df = data[data['compressor'] == comp.lower()] #pick the compressor
        compressors_dict[comp] = {}

        for idx, bound in enumerate(bounds):
            
            compressors_dict[comp][error_bounds[idx]] = {}
            comp_error_df = comp_df[comp_df['bound'].apply(isclose, args=[bound]) ]
            
            mean_cr = (comp_error_df['uncompressed_size_bytes']/comp_error_df['compressed_size_bytes']).mean()
            mean_psnr = (comp_error_df['psnr']).mean()
            mean_ssim = (comp_error_df['ssim']).mean()
            total_time = comp_error_df['total_time'].sum()
            compressed_size_bytes = comp_error_df['compressed_size_bytes'].sum()
            uncompressed_size_bytes = comp_error_df['uncompressed_size_bytes'].sum() #sum to treat it like total size for dataset

            compressors_dict[comp][error_bounds[idx]]["cr"] = mean_cr
            compressors_dict[comp][error_bounds[idx]]["psnr"] = mean_psnr
            compressors_dict[comp][error_bounds[idx]]["ssim"] = mean_ssim
            compressors_dict[comp][error_bounds[idx]]["total_time"] = total_time
            compressors_dict[comp][error_bounds[idx]]["compressed_size_bytes"] = compressed_size_bytes 
            compressors_dict[comp][error_bounds[idx]]["uncompressed_size_bytes"] = uncompressed_size_bytes

            print(f"num images is: {comp_error_df.shape}")
            print(f"Error Bound: {error_bounds[idx]}")
            print(f"The mean CR for rows with compressor '{comp}' is: {mean_cr}")
            print(f"The mean psnr for rows with compressor '{comp}' is: {mean_psnr}")
            print(f"The mean ssim for rows with compressor '{comp}' is: {mean_ssim}")
            print(f"The time taken for rows with compressor '{comp}' is: {total_time} seconds")
            print(f"The size it compressed it to with compressor '{comp}' is: {compressed_size_bytes} bytes")
            print(f"Uncompressed size is is: {uncompressed_size_bytes} bytes")
    else if comp == "JPEG":
        jpeg_comp_df = jpeg_data[jpeg_data["split"] == 0]
        compressors_dict[comp] = {}
        
        for idx, quality in enumerate(quality_levels):
            compressors_dict[comp][quality] = {}
            jpeg_quality_df = jpeg_comp_df[jpeg_comp_df['quality'] == quality]
            
            mean_cr = (jpeg_quality_df['uncompressed_size_bytes']/jpeg_quality_df['compressed_size_bytes']).mean()
            mean_psnr = (jpeg_quality_df['psnr']).mean()
            mean_ssim = (jpeg_quality_df['ssim']).mean()
            total_time = jpeg_quality_df['total_time'].sum()
            compressed_size_bytes = jpeg_quality_df['compressed_size_bytes'].sum()
            uncompressed_size_bytes = jpeg_quality_df['uncompressed_size_bytes'].sum() #sum to treat it like total size for dataset
            compressors_dict[comp][quality]["cr"] = mean_cr
            compressors_dict[comp][quality]["psnr"] = mean_psnr
            compressors_dict[comp][quality]["ssim"] = mean_ssim
            compressors_dict[comp][quality]["total_time"] = total_time
            compressors_dict[comp][quality]["compressed_size_bytes"] = compressed_size_bytes 
            compressors_dict[comp][quality]["uncompressed_size_bytes"] = uncompressed_size_bytes

            print(f"num images is: {jpeg_quality_df.shape}")
            print(f"Quality level: {quality}")
            print(f"The mean CR for rows with compressor '{comp}' is: {mean_cr}")
            print(f"The mean psnr for rows with compressor '{comp}' is: {mean_psnr}")
            print(f"The mean ssim for rows with compressor '{comp}' is: {mean_ssim}")
            print(f"The time taken for rows with compressor '{comp}' is: {total_time} seconds")
            print(f"The size it compressed it to with compressor '{comp}' is: {compressed_size_bytes} bytes")
            print(f"Uncompressed size is is: {uncompressed_size_bytes} bytes")


#in MiB per second
bandwidths = [1000, 100, 50, 10, 5,] #lets say this is in Mb per second

compressors = ['SZ3', 'ZFP', 'JPEG', 'Uncompressed']

compressor_labels = []

error_bounds = ['1E-7', '1E-6', '1E-5', '1E-4', '1E-3', '1E-2', '1E-1']
quality_levels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

for comp in compressors:
    if comp != 'Uncompressed' and comp != "JPEG":
        for error_bound in error_bounds:
            label = '{} CR:{:0.2f}'.format(comp, compressors_dict[comp][error_bound]["cr"])
    else if comp == "JPEG":
        for quality in quality_levels:
            label = '{} CR:{:0.2f}'.format(comp, compressors_dict[comp][quality]["cr"])
    else:
        label = "Uncompressed"
    compressor_labels.append(label)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (replace with your actual data)
#compression_ratios = np.linspace(1, 10, 20)  # From 1:1 to 10:1
#compressors = ['zstd','blosclz','lz4','lz4hc','zlib'] #all are lossless except for zfp and sz3 which are lossy (for now lets only include lossless)

# Create a 2D array for the heat map values
# This example uses random data - replace with your actual measurements
#These are the times it takes for that configuration to theoretically send
data = np.random.rand(len(bandwidths), len(compressor_labels))

#time_to_compress + compressed_dataset_size / (transfer bandwidth) + time_to_decompress

#compressed_ds_size = ogsize / cr
#comp_time_avg + (compress_ds_size / bandwidths[i]) + decomp_time_avg OR (compress_ds_size / bandwidths[i]) + total_time_to_compress_and_decompress

#although I got a mismatch in uncompressed_ds_size for libpressio and for doing os.path.getsize, for consistency libpressio's numbers will be used

#converted to megabytes
uncompressed_ds_size = compressors_dict[compressors[0]]['1E-1']["uncompressed_size_bytes"] / 1000000

print(uncompressed_ds_size)

for idx_i, comp in enumerate(compressors):
    #compressed_ds_size = original_ds_size / compressors_dict[comp]["cr"] #og_ds_size / cr
    if comp != 'Uncompressed' and comp != "JPEG":
        #divide by 1000000 to convert it from bytes to megabytes (mb is 1e6 bytes)
        compressed_ds_size = compressors_dict[comp]['1E-1']["compressed_size_bytes"] / 1000000
    else if comp == "JPEG":
        compressed_ds_size = compressors_dict[comp]['0']["compressed_size_bytes"] / 1000000
    
    for idx_j, bw in enumerate(bandwidths):
        if comp == "JPEG":
            #for compressors
            data[idx_j][idx_i] = (compressed_ds_size / bw) + compressors_dict[comp]['0']["total_time"]
        else if comp != 'Uncompressed' and comp != "JPEG":
            #for compressors
            data[idx_j][idx_i] = (compressed_ds_size / bw) + compressors_dict[comp]['1E-1']["total_time"]
        else:
            #for without a compressor
            data[idx_j][idx_i] = (uncompressed_ds_size / bw)


# Set the global font to Times New Roman
import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman']


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Increase the font sizes
SMALL_SIZE = 12*1.3
MEDIUM_SIZE = 14*1.3
BIGGER_SIZE = 16*1.3

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Create the heat map
fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure size
sns.heatmap(data, xticklabels=compressor_labels, yticklabels=bandwidths,
            cmap='YlOrRd', fmt='.2f', annot=True, ax=ax, cbar=True, 
            cbar_kws={'label': 'Time (seconds)'}, annot_kws={"size": SMALL_SIZE})

# Function to determine text color based on background
def text_color_for_background(value):
    threshold = (data.max() - data.min()) / 2.0 + data.min()
    return 'white' if value > threshold else 'black'

# Manually add text with conditional coloring
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        value = data[i, j]
        text_color = text_color_for_background(value)
        ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
                ha="center", va="center", color=text_color, fontsize=SMALL_SIZE)

plt.xlabel('Compressor', fontsize=MEDIUM_SIZE)
plt.ylabel('Bandwidth (Mb/s)', fontsize=MEDIUM_SIZE)
plt.title('Compression Performance Heat Map', fontsize=BIGGER_SIZE)

# Rotate and align the tick labels so they look better
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Adjust layout
plt.tight_layout()

# Save the figure as PNG
plt.savefig('lossy_heatmap_final_v2.png', dpi=500, bbox_inches='tight')

# If you still want to display the plot, keep this line
plt.show()

































