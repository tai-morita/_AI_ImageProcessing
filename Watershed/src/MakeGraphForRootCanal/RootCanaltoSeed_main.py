# WaterShed の Seed を Root Canal にする
"""
input[1]: CTHRs_100_Labeln.npy -> 元のボリュームデータ
input[2]: Labeln.npy           -> 歯を抽出した 2 値化データ
return  : ToothSplitted_CTHRs_100_Labeln.tif -> WaterShed 結果データ

中間画像
1. CTHRs_100_Labeln_overlay.tif -> 元ボリュームデータと抽出データのオーバーレイ画像
2. CTHRs_100_Labeln_RC2_T1_B0.tif -> 歯が 1, 根管が 2, 背景が 0 のラベル画像
3. watershed_Tooth_label_conn=26_color.tif

処理
1. 元ボリュームデータと抽出データから、ボリュームの抽出データを作成する
    中間保存: CTHRs_100_Labeln_overlay.tif, CTHRs_100_Labeln_overlay.npy
    return: CTHRs_100_Labeln_overlay: np.ndarray
2. 2 値化処理で根幹を抽出する
    return: CTHRs_100_Labeln_root_canal: np.ndarray
3. 根管のラベル付けをし、外れ値 (ノイズ) は除去する
    - 周囲 5 ボクセルは同一ラベルとして扱う
    - ラベルごとのサイズ分布に大きな飛びがある箇所を境界として外れ値を除去する
    return: CTHRs_100_Labeln_root_canal_labeled: np.ndarray
4. CTHRs_100_Labeln_root_canal_labeled を Seed として Watershed を実行する
    return: ToothSplitted_CTHRs_100_Labeln.tif
"""

