import os
import numpy as np
import pandas as pd
import rasterio  # Handles multi-band images

# Path to EuroSAT dataset (update this to your dataset location)
data_dir = "/Users/maianhpham/Desktop/tif"

# List all class folders (ignore non-directory files)
classes = [c for c in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, c))]

# Initialize list for storing data
data_list = []

# Iterate through classes
for class_name in classes:
    class_path = os.path.join(data_dir, class_name)

    # Process images in each class folder
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Ignore non-TIF files
        if not img_name.endswith(".tif"):
            continue

        # Open the image and extract all 13 bands
        with rasterio.open(img_path) as src:
            bands = [src.read(band).mean() for band in range(1, 14)]  # Read all 13 bands

        # Append data: Bands + Class Label
        data_list.append([*bands, class_name])

# Convert to DataFrame
columns = [f"Band{i}" for i in range(1, 14)] + ["Label"]
df = pd.DataFrame(data_list, columns=columns)

# Save as CSV
csv_output = "EuroSAT_13bands.csv"
df.to_csv(csv_output, index=False)

print(f" CSV file '{csv_output}' generated successfully with 13 bands and labels!")