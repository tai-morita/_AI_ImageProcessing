import numpy as np
import tifffile as tiff
import csv

input = r"./study/Watershed/t-oe/20260716_NR_Label/label_1/CTHRs_100_Label1_seed_root_canal.tif"
volume = tiff.imread(input)

# ラベル数を確認
unique_labels = np.unique(volume)

output_csv = f"./temp.csv"
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Label', 'Slice Index', 'Area'])
for label in unique_labels:
    if label == 0:
        continue  # 背景ラベルはスキップ

    for slice_index in range(volume.shape[0]):
        slice_mask = volume[slice_index] == label
        if np.any(slice_mask):
            # ラベルが存在するスライスの面積を計算
            area = np.sum(slice_mask)
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([label, slice_index, area])