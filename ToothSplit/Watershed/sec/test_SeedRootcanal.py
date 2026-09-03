# Root Canal を Seed にして Watershed を実行するテスト
import numpy as np
from scipy import ndimage as ndi

def edit_root_canal(labeled_root_canal: np.ndarray, tooth_labeled_volume: np.ndarray) -> np.ndarray:
    """
    ラベル付きの根管領域を編集する。
    0 を背景として除外し、ラベルごとのサイズ分布に大きな飛びがある箇所を境界として外れ値を除去する。
    """
    unique_labels, counts = np.unique(labeled_root_canal, return_counts=True)

    # 背景 0 は除外
    mask = unique_labels != 0
    unique_labels = unique_labels[mask]
    counts = counts[mask]

    if unique_labels.size == 0:
        return labeled_root_canal

    # サイズを昇順にソート
    order = np.argsort(counts)
    unique_labels = unique_labels[order]
    counts = counts[order]

    # 対数変換して、急なジャンプを見つける
    log_counts = np.log10(counts.astype(np.float64))
    jumps = np.diff(log_counts)

    if jumps.size == 0:
        return labeled_root_canal

    # 最大ジャンプ位置を境界として採用
    boundary_idx = np.argmax(jumps)
    threshold = counts[boundary_idx + 1]

    # 閾値未満のラベルは背景化
    valid_labels = unique_labels[counts < threshold]

    # ラベルが枠外に出ている、つまり 0, 1 or 最大スライス存在している or 背景に接しているラベルは対象外とする
    adjacent_background_list = check_adjacent_root_background(labeled_root_canal, tooth_labeled_volume)
    adjant_axial_list = check_adjacent_root_axial(labeled_root_canal)
    exception_edit_root_canal_labels = list(set(adjacent_background_list + adjant_axial_list))

    edit_labels = labeled_root_canal.copy()
    # print(f"編集対象のラベル: {valid_labels}, 編集対象外のラベル: {exception_edit_root_canal_labels}")
    for edit_label in valid_labels:
        if edit_label in exception_edit_root_canal_labels:
            continue  # 編集対象外のラベルはスキップ
        edit_labels[edit_labels == edit_label] = 0  # 背景化

    # 周囲 5 ボクセルを膨張させてラベルを再付与する
    from .LabeledRootCanal import relabel_nearby_labeled_regions
    edit_labels = relabel_nearby_labeled_regions(edit_labels, dilation_iterations=5)

    # 編集前と編集後のラベルとそのカウントを表示する
    if False:
        print(f"Max: {np.max(labeled_root_canal)} -> {np.max(edit_labels)}")
    
    return edit_labels

# ラベル数が 0 のものを除外して、ラベルを 1 から順に振り直す
def relabel_without_zero(arr: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(arr)
    unique_labels = unique_labels[unique_labels != 0]  # 0 を除外

    for new_label, old_label in enumerate(unique_labels, start=1):
        arr[arr == old_label] = new_label

    return arr

def check_adjacent_root_axial(labeled_root_canal: np.ndarray) -> list:
    """
    Axial 方向で見切れているか判定する。 0, 1, max-1, max のスライスで根管ラベルが背景と接触している場合は見切れていると判定する。
    Parameters:
        labeled_root_canal (np.ndarray): ラベル付きの根管領域
    Returns:
        list: 背景と接触している根管ラベルのリスト
    """
    slice_indexes_to_check = [0, 1, labeled_root_canal.shape[0] - 2, labeled_root_canal.shape[0] - 1]
    contacting_root_numbers = []
    for slice_index in slice_indexes_to_check:
        # このスライスにラベルがあれば接触対象となる
        labeled_numbers = np.unique(labeled_root_canal[slice_index])
        labeled_numbers = labeled_numbers[labeled_numbers != 0]  # 背景は除外
        for root_number in labeled_numbers:
            contacting_root_numbers.append(int(root_number))
    contacting_root_numbers = list(set(contacting_root_numbers))  # 重複を除外
    return contacting_root_numbers

def check_adjacent_root_background(
    labeled_root_canal: np.ndarray,
    tooth_labeled_volume: np.ndarray
    ) -> list:
    """
    根管ラベルが背景と隣接しているかどうかを判定する。

    Parameters:
        labeled_root_canal (np.ndarray): ラベル付きの根管領域
        tooth_labeled_volume (np.ndarray): ラベル付きの歯領域

    Returns:
        list: 背景と接触している根管ラベルのリスト
    """
    root_mask = labeled_root_canal > 0
    root_numbers = np.unique(labeled_root_canal[root_mask])
    background_mask = tooth_labeled_volume == 0

    # 6近傍: z, y, x の各軸に直交する隣接ボクセルだけを対象にする
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)

    contacting_root_numbers = []
    for root_number in root_numbers:
        root_mask = labeled_root_canal == root_number
        # root_mask を1ボクセルだけ膨張し、背景と重なる場所を得る
        adjacent_background = (
            ndi.binary_dilation(root_mask, structure=structure)
            & background_mask
        )

        if np.any(adjacent_background):
            contacting_root_numbers.append(int(root_number))
            break  # 1つでも接触していれば十分なので、ループを抜ける

    return contacting_root_numbers

def connected_component(slice: np.ndarray, labeled_slice: np.ndarray, target_value: int) -> np.uint16:
    """
    labeled_slice 内の target_value に対応する slice の Connected Component の面積を返す。

    Parameters:
        slice (np.ndarray): 2D スライス画像 (2 値化画像)
        labeled_slice (np.ndarray): 2D ラベル付きスライス画像
        target_value (int): 対象のラベル値
    Returns:
        np.uint16: target_value に対応する Connected Component の面積
    """
    # target_value 以外のラベルは除外して、2 値化画像を作成
    target_mask = labeled_slice == target_value

    if not np.any(target_mask):
        print(f"ラベル {target_value} はスライス内に存在しません。")
        return np.uint16(0)

    target_intensity = slice[target_mask][0]  # target_value に対応する slice の値を取得
    foreground_mask = slice == target_intensity

    # 2Dの4近傍で slice の連結成分を抽出
    structure = ndi.generate_binary_structure(rank=2, connectivity=1)
    component_labels, _ = ndi.label(foreground_mask, structure=structure)

    # target_value が重なる連結成分番号を取得
    # target_mask は target_value の場所のみ True で、それ以外は False
    component_ids = np.unique(component_labels[target_mask])
    component_ids = component_ids[component_ids != 0]

    component_mask = np.isin(component_labels, component_ids)
    return np.uint16(np.count_nonzero(component_mask))

def get_volume_component_ids(
    labeled_volume: np.ndarray,
    labeled_root_canal: np.ndarray,
    root_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    root_number の根管ラベルを含む、volume の 3D Connected Component 番号を返す。

    Parameters:
        labeled_volume (np.ndarray): ラベル付きの volume
        labeled_root_canal (np.ndarray): ラベル付きの根管領域
        root_number (int): 根管ラベル番号
    Returns:
        volume_labels: volume の 3D Connected Component ラベル画像
        component_ids: root_number と重なる Connected Component 番号の配列
    """

    root_mask = labeled_root_canal == root_number
    component_ids = np.unique(labeled_volume[root_mask])

    # 背景の Connected Component 番号 0 は除外
    component_ids = component_ids[component_ids != 0]

    return labeled_volume, component_ids

# 根管が見切れている場合、歯冠部までラベル領域を増やす
def expand_root_canal_to_crown(labeled_root_canal: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    根管ラベルが見切れている場合、歯冠部までラベル領域を増やす。
    根管ラベルに含まれる Connected Component の面積を計算し、面積増加量が非常に多いスライスに達するまでラベル領域を増やす。

    見切れる判定: 根管と背景が隣接している

    Parameters:
        labeled_root_canal (np.ndarray): ラベル付きの根管領域
        volume (np.ndarray): 元のボリュームデータ (2 値化画像でよい)
    Returns:
        np.ndarray: 編集後のラベル付き根管領域
    """

    # 根管が背景と接触しているラベルを取得
    contacting_root_numbers = check_adjacent_root_background(labeled_root_canal)

    for root_number in contacting_root_numbers:
        # 根管ラベルの含まれる Connected Component のうち、一番面積の小さいスライスを見つける
        root_mask = labeled_root_canal == root_number
        slice_areas = np.array([np.sum(root_mask[z, :, :]) for z in range(root_mask.shape[0])])
        min_area_slice = np.argmin(slice_areas)
        max_area_slice = np.argmax(slice_areas)
        # スライスを走査する方向を決める。
        # max - min < 0 -> 下方向に捜査する
        # max - min > 0 -> 上方向に捜査する
        direction = -1 if max_area_slice < min_area_slice else 1

        slice_index = min_area_slice
        while True:
            # 現在のスライスの面積を計算
            current_area = connected_component(volume[slice_index, :, :], labeled_root_canal[slice_index, :, :], root_number)

            # 次のスライスに進む
            slice_index += direction

            # スライスが範囲外になったら終了
            if slice_index < 0 or slice_index >= volume.shape[0]:
                break

            # 次のスライスの面積を計算
            next_area = connected_component(volume[slice_index, :, :], labeled_root_canal[slice_index, :, :], root_number)

            # 面積増加量が非常に多い場合、ラベル領域を増やす
            if next_area > current_area * 1.3:  # 面積が 1.3 倍以上になったら終了
                # ラベル領域を増やす
                labeled_root_canal[slice_index, :, :][volume[slice_index, :, :] > 0] = root_number
                break

    return labeled_root_canal

def expand_seed_volume(labeled_root_canal: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    根管ラベルが見切れている場合、歯冠部までラベル領域を増やす。
    根管ラベルに含まれる Connected Component の面積を計算し、面積増加量が非常に多いスライスに達するまでラベル領域を増やす。

    処理内容
    1. 根管ラベルのうち、背景と接触しているラベル(見切れたラベル)を取得する (check_adjacent_root_background)
    2. 見切れたラベルごとに、根管ラベルの含まれる Connected Component の歯が上向きか下向きかを判定する (未定)
    3. Volume の Connected Component を抽出する (ndi.label)
    4. 見切れた根管ラベルを含む Volume の Connected Component を抽出する (get_component_label)
    5. 歯冠部方向にスライスを走査し、 Connected Component の面積を計算する (connected_component)
    6. 面積増加量が非常に多いスライスに達するまでラベル領域を増やす (expand_root_canal_to_crown)

    Parameters:
        labeled_root_canal (np.ndarray): ラベル付きの根管領域
        volume (np.ndarray): 元のボリュームデータ (2 値化画像でよい)
    Returns:
        np.ndarray: 編集後のラベル付き根管領域
    """

    # Step1. 根管ラベルのうち、背景と接触しているラベル(見切れたラベル)を取得する (check_adjacent_root_background)
    contacting_root_numbers = check_adjacent_root_background(labeled_root_canal)

    # Step2. 見切れたラベルごとに、根管ラベルの含まれる Connected Component の歯が上向きか下向きかを判定する (未定)

    # Step3. Volume の Connected Component を抽出する (ndi.label)
    foreground_mask = volume > 0
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    labeled_volume, _ = ndi.label(foreground_mask, structure=structure)

    for root_number in contacting_root_numbers:
        # Step4. 見切れた根管ラベルを含む Volume の Connected Component を抽出する (get_component_label)
        labeled_volume, component_id = get_volume_component_ids(labeled_volume, labeled_root_canal, root_number)
        if component_id == 0:
            continue  # 対応する Connected Component が見つからなかった場合はスキップ

        while True:
            # Step5. 歯冠部方向にスライスを走査し、 Connected Component の面積を計算する (connected_component)
            

            # Step6. 面積増加量が非常に多いスライスに達するまでラベル領域を増やす (expand_root_canal_to_crown)
            labeled_root_canal = expand_root_canal_to_crown(labeled_root_canal, volume, component_id)

    return labeled_root_canal

if __name__ == "__main__":
    volume = np.array([[
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,1
    ]])

    label = np.array([[
        0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0,0,
        0, 4, 4, 4, 0, 0, 0, 0, 0, 0,1
    ]])

    area = connected_component(volume, label, target_value=3)

    print(area)             # 現状は 21