#!/bin/bash
# Start counter
count=1

# Loop through files in sorted order
for file in Brown_Field/Test/imgs/img_*.png; do
    # Format counter with leading zeros (3 digits)
    new_name=$(printf "frame%03d.png" $count)
    # Rename the file
    mv "$file" "$new_name"
    # Increment counter
    count=$((count + 1))
done
