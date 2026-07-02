from ..pipeline import GraphRev4Config, graph_main_anterior_molar_pca_rev4


def run_legacy_style(
    input_path: str,
    output_dir: str,
    threshold: int = 1000,
    anterior_threshold: int | None = None,
    molar_threshold: int | None = None,
    labels_connectivity: int = 2,
    direction_mode: str | None = None,
    execution_slice_range: tuple | None = None,
    run_watershed: bool = True,
):
    """旧来スタイルの引数でrev4パイプラインを呼び出す互換関数。

    Args:
        input_path: 入力ボリュームのパス。
        output_dir: 出力ディレクトリ。
        threshold: 共通面積閾値。
        anterior_threshold: 前歯領域の閾値。
        molar_threshold: 臼歯領域の閾値。
        labels_connectivity: 2D連結成分の近傍設定。
        direction_mode: forward_only / reverse_only / bidirectional。
        execution_slice_range: 実行対象の1-basedスライス範囲。
        run_watershed: watershed実行の有無。

    Returns:
        統合seedボリュームと実行情報辞書。
    """
    config = GraphRev4Config(
        input_path=input_path,
        output_dir=output_dir,
        threshold=threshold,
        anterior_threshold=anterior_threshold,
        molar_threshold=molar_threshold,
        labels_connectivity=labels_connectivity,
        direction_mode=direction_mode,
        execution_slice_range=execution_slice_range,
        run_watershed=run_watershed,
    )
    return graph_main_anterior_molar_pca_rev4(config)
