# DataLoadTest notebook -> Python module

このディレクトリは、`DataLoadTest.ipynb` の処理を保守しやすい形に分割した実装です。

## ファイル構成
- `settings.py`: 実行設定 (`DataLoadConfig`)
- `client.py`: API 通信と認証付きリクエスト
- `raw_volume.py`: RAW ボリュームのデコード、回転、コントラスト計算
- `labels.py`: ラベル npz の読込
- `visualization.py`: 可視化補助
- `pipeline.py`: 一連処理の統合 (`run_pipeline`)
- `main.py`: 実行エントリ

## 実行例
`study/test/src` をカレントにして:

```powershell
python -m dataload_pipeline.main
```

## メンテナンス方針
- API仕様の変更は `client.py` のみに閉じ込める
- RAWフォーマット変更は `raw_volume.py` に集約する
- 表示ロジックは `visualization.py` に限定する
