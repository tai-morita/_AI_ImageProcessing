from __future__ import annotations
from typing import Literal, Tuple, Dict, Any
import numpy as np
from skimage import io, color, exposure, filters, morphology, util
from scipy import ndimage as ndi
import os
import numpy as np

if __name__ == "__main__":
    # 画像2つの差分を撮って保存する
    # いい感じにやりたいから２値化する
    img1 = io.imread(r"./study/test/Input/water_coins.jpg", as_gray=True)
    img2 = io.imread(r"./study/test/Input/marked_coin_gray.png", as_gray=True)
    diff = np.abs(img1 - img2)
    threshold = filters.threshold_otsu(diff)
    diff_binary = diff > threshold
    save_path = './study/test/Output/diff.png'
    io.imsave(save_path, (diff_binary * 255).astype(np.uint8))
    np.savetxt('./study/test/Output/diff.csv', diff_binary, delimiter=',', fmt='%d')