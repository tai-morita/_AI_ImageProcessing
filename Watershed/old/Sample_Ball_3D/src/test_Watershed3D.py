import numpy as np
from collections import deque
from typing import Tuple, Literal

def test_watershed_vincent_soille(
    image: np.ndarray,
    connectivity: Literal[4, 8] = 8,
    keep_watershed: bool = True,
) -> np.ndarray:
    """
    Vincent & Soille (1991) の提案手法に基づく Watershed 実装（2D）。
    - 入力: 2D グレースケール画像（整数型推奨; 負値も可）
    - 出力: ラベル画像（int32）
        * 1..K : 各 catchment basin のラベル
        * 0     : WSHED（keep_watershed=True の場合）
      keep_watershed=False の場合は、WHS を近傍ラベルで埋めてタイル分割にします（簡便な後処理）。

    アルゴリズム上の定数:
        INIT = -1, MASK = -2, WSHED = 0

    注意:
      - 画像境界は自然に処理（領域外チェック）
      - connectivty: 4 or 8
      - メモリ: O(N). 時間: 事実上 O(N)

    参考: Vincent & Soille, PAMI 1991
    """
    # ---- 入力検証 ----
    if image.ndim != 2:
        raise ValueError("image は 2次元配列である必要があります。")
    H, W = image.shape

    # 整数型へ（内部は int64 で処理）
    img = np.asarray(image, dtype=np.int64)
    N = H * W

    # ---- 近傍定義 ----
    # 4 or 8 近傍の相対座標
    if connectivity == 4:
        neighbor_rc = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbor_rc = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        raise ValueError("connectivity は 4 または 8 を指定してください。")

    # 線形インデックス <-> (r, c) の簡便関数
    def lin2rc(idx: int) -> Tuple[int, int]:
        return idx // W, idx % W

    def neighbors_lin(idx: int):
        r, c = lin2rc(idx)
        for dr, dc in neighbor_rc:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr * W + cc

    # ---- 出力 / 作業配列 ----
    INIT, MASK, WSHED = -1, -2, 0
    labels = np.full(N, INIT, dtype=np.int32)       # 出力ラベル（線形）
    dist   = np.zeros(N, dtype=np.int16)            # 作業距離（層管理; 2bitでも可だが単純化）

    # ---- 計数ソート（濃度レベルごとの直接アクセスを確保）----
    vmin, vmax = int(img.min()), int(img.max())
    offset = -vmin
    # 負値ありでもOK
    hist = np.bincount((img.ravel() + offset), minlength=(vmax - vmin + 1))
    # cumulative[i] = レベル vmin + i の開始位置
    cumulative = np.zeros(hist.size + 1, dtype=np.int64)
    np.cumsum(hist, out=cumulative[1:])

    # level-sorted な線形インデックス配列 indices を構築
    indices = np.empty(N, dtype=np.int64)
    # 作業ポインタ
    write_ptr = cumulative.copy()
    flat = (img.ravel() + offset)
    for idx_lin in range(N):
        lvl = flat[idx_lin]
        pos = write_ptr[lvl]
        indices[pos] = idx_lin
        write_ptr[lvl] += 1

    # ---- レベルごとの浸水 ----
    current_label = 0
    q = deque()

    for lvl in range(hist.size):  # lvl=0..(vmax-vmin)
        # レベルに属する線形インデックス範囲 [start:end)
        start, end = cumulative[lvl], cumulative[lvl + 1]
        if start == end:
            continue  # このレベルは画素無し

        # 1) レベル h の画素を MASK にセット
        for k in range(start, end):
            p = indices[k]
            labels[p] = MASK

        # 2) 既ラベル or WSHED に隣接する MASK を queue に入れる (imd=1)
        for k in range(start, end):
            p = indices[k]
            for qn in neighbors_lin(p):
                if labels[qn] > 0 or labels[qn] == WSHED:
                    dist[p] = 1
                    q.append(p)
                    break  # p は queue 済み

        # 3) FIFO BFS で geodesic influence zone を拡張
        if q:
            q.append(None)  # fictitious pixel（距離層の境界）
            curdist = 1
            while q:
                p = q.popleft()
                if p is None:
                    # 距離層境界
                    if not q:
                        break
                    q.append(None)
                    curdist += 1
                    continue

                # p の近傍を巡回
                for qn in neighbors_lin(p):
                    # 近傍が既ラベル or WSHED で、より“内側（<= curdist-1層）”
                    if dist[qn] < curdist and (labels[qn] > 0 or labels[qn] == WSHED):
                        if labels[qn] > 0:
                            if labels[p] == MASK or labels[p] == WSHED:
                                labels[p] = labels[qn]
                            elif labels[p] != labels[qn]:
                                labels[p] = WSHED
                        elif labels[qn] == WSHED:
                            if labels[p] == MASK:
                                labels[p] = WSHED

                    # 近傍が未処理の MASK なら、次層として距離設定＋enqueue
                    elif labels[qn] == MASK and dist[qn] == 0:
                        dist[qn] = curdist + 1
                        q.append(qn)

            # 次レベルに向けて、dist をこのレベル分だけクリア（後でまとめてやるためフラグ管理も可）
            # ここでは簡単のため、当該レベル画素だけ0に戻すのを後段 4) のループで行う

        # 4) 新たな極小（まだ MASK のまま）を新規ラベルで塗りつぶし
        for k in range(start, end):
            p = indices[k]
            if dist[p] != 0:
                dist[p] = 0  # 当レベルの dist をクリア
            if labels[p] == MASK:
                current_label += 1
                labels[p] = current_label
                q.append(p)
                while q:
                    x = q.popleft()
                    for y in neighbors_lin(x):
                        if labels[y] == MASK:
                            labels[y] = current_label
                            q.append(y)

    labels_2d = labels.reshape(H, W)

    # ---- オプション後処理：WSHED を近傍の最小ラベルで埋めてタイル分割にする ----
    if not keep_watershed:
        # 1パスで埋まらない場合もあるので、簡易に数回繰り返す
        # （厳密な“厚い”WSHED の扱いは用途依存。ここでは簡便法。）
        for _ in range(4):
            ws_mask = (labels_2d == WSHED)
            if not ws_mask.any():
                break
            # 近傍の正ラベルの最小を当てる
            fill = np.full_like(labels_2d, 0)
            for r in range(H):
                for c in range(W):
                    if labels_2d[r, c] == WSHED:
                        neigh = []
                        for dr, dc in neighbor_rc:
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < H and 0 <= cc < W:
                                v = labels_2d[rr, cc]
                                if v > 0:
                                    neigh.append(v)
                        if neigh:
                            fill[r, c] = min(neigh)
            labels_2d[ws_mask & (fill > 0)] = fill[ws_mask & (fill > 0)]

    return labels_2d