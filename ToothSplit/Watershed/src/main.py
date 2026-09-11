
import os
import numpy as np
from SplitTeethMain import SplitTeethMain
from DataLoad import load_data, save_volume
def main():
    label_numbers = [1, 2, 3]
    for label_number in label_numbers:
        input_volume_path       = fr"D:\_study\_AI_ImageProcessing\ToothSplit\Watershed\data\label_{label_number}\CTHRs_100_Label{label_number}.npy"
        input_teeth_binary_path = fr"D:\_study\_AI_ImageProcessing\ToothSplit\Watershed\data\label_{label_number}\Label{label_number}.npy"
        extracted_volume, edited_labeled_root_canal, splitted_teeth_volume = SplitTeethMain(input_volume_path, input_teeth_binary_path)
        # 結果の保存
        # output_dir = r"D:\_study\_AI_ImageProcessing\ToothSplit\Watershed\data\temp"
        output_dir = fr"D:\_study\_AI_ImageProcessing\ToothSplit\Watershed\data\label_{label_number}"
        if True:
            save_volume(os.path.join(output_dir, f"CTHRs_100_Label{label_number}_extracted_volume.npy")         , extracted_volume)
            save_volume(os.path.join(output_dir, f"CTHRs_100_Label{label_number}_labeled_root_canal.npy"), edited_labeled_root_canal, dtype = np.int8)
            save_volume(os.path.join(output_dir, f"CTHRs_100_Label{label_number}_splitted_teeth.npy")    , splitted_teeth_volume    , dtype = np.int8)
        if True:
            save_volume(os.path.join(output_dir, f"CTHRs_100_Label{label_number}.tif")      , load_data(os.path.join(output_dir, input_volume_path))      , dtype = np.float32)
            save_volume(os.path.join(output_dir, f"Label{label_number}.tif"), load_data(os.path.join(output_dir, input_teeth_binary_path)), dtype = np.int8)
            save_volume(os.path.join(output_dir, f"CTHRs_100_Label{label_number}_extracted_volume.tif")         , extracted_volume)
            save_volume(os.path.join(output_dir, f"CTHRs_100_Label{label_number}_labeled_root_canal.tif"), edited_labeled_root_canal, dtype = np.int8)
            save_volume(os.path.join(output_dir, f"CTHRs_100_Label{label_number}_splitted_teeth.tif")    , splitted_teeth_volume)

if __name__ == "__main__":
    main()