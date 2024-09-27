#!/bin/bash

# Directory containing input frames
input_dir="Brown_Field/Test/imgs"

# Output video file
output_file="output_compressed.mp4"

# Frame rate of the output video
frame_rate=30

# CRF value (lower = higher quality, higher file size)
crf=23

# Preset (slower presets = better compression, but longer encoding time)
preset="medium"

# Compress frames to H.265 video
ffmpeg -framerate $frame_rate -pattern_type glob -i "$input_dir/*.png" \
       -c:v libx265 -preset $preset -crf $crf \
       -pix_fmt yuv420p \
       $output_file

echo "Compression completed. Output saved as $output_file"