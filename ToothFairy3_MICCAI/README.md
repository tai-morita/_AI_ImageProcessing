# nnU-Net を勉強する
## 概要
- 歯の分割 AI モデル開発のため、 nnU-Net について勉強する。
- 以下の Tooth fairy 3 は MICCAI Challenge で、歯などのデータセットと、モデルがたくさんある。それを参考にモデル開発をしてみる。
## Tooth fairy 3 サイト
https://toothfairy3.grand-challenge.org/toothfairy3/
## フォルダ構成
```
ToothFairy3_MICCAI/
├── ToothFairy3.zip
├── README.md
├── docs/
├── scripts/
│   └── remap_to_nnunet_consecutive.py--------- ToothFairy3 のラベル番号振り直しの実行ファイル
├── nnUNet_raw/ ------------------------------- nnU-Net が読む raw データ
│   ├── Dataset501_ToothFairy3/ --------------- ToothFairy3 の学習用元データ. ラベル番号が適していないので, nnU-Net の学習にそのまま使えない. (21, 31, 103, 111 など)
│   └── Dataset502_ToothFairy3_consecutive/ --- nnU-Net 用にラベル番号を振り直した学習用データ. (0, 1, 2, ...)これを使って学習する.
├── nnUNet_preprocessed/                    --- nnU-Net が前処理したデータを置く場所.
│   └── Dataset502_ToothFairy3_consecutive/ ---
└── nnUNet_results/                         --- 学習結果が入る.
```
## 学習準備
- docker コンテナでの作業する場合
1. /workspace へ移動
```
cd /workspace
```
2. コンテナの環境変数を設定する (コンテナを起動するたびに実行する)
```
export nnUNet_raw=/workspace/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnUNet_preprocessed
export nnUNet_results=/workspace/nnUNet_results
```
確認
```
echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results
```
想定結果
```
/workspace/nnUNet_raw
/workspace/nnUNet_preprocessed
/workspace/nnUNet_results
```
## 学習方法
1. 学習開始
コマンド例
```
nnUNetv2_train 502 2d 0 -device cuda 2>&1 | tee train_502_2d_fold0.log
```
- nnUNetv2_train
  - python 内の nnunetv2 モジュール実行
- 502
  - 学習データとして Dataset502_ToothFairy3_consecutive を使う
- 2d
  - 2D U-Net で学習する
- 0
  - fold 0 を学習する
- -device cuda
  - GPU を使用する
2. 学習中・学習後
学習結果は以下に保存される
```
nnUNet_results/Dataset502_ToothFairy3_consecutive/
```
2d, fold 0 学習後のフォルダ構成.
```
└── Dataset502_ToothFairy3_consecutive/
    └── nnUNetTrainer__nnUNetPlans__2d/
        └── fold_0/
            ├── checkpoint_best.pth   --- 学習後のモデル(中間?)
            ├── checkpoint_final.pth  --- 学習後のモデル
            ├── progress.png          --- 学習曲線. loss が下がっているかを確認する.
            ├── training_log_XXXX.txt
            └── validation/
                └──summary.json       ---
```
## その他
### 学習を途中で止める場合
止めればよい
```
Ctrl + C
```
### 学習が途中で止まった場合
--c は Continue の意味. 再開する.
```
nnUNetv2_train 502 2d 0 -device cuda --c 2>&1 | tee train_502_2d_fold0.log
```
中間が保存されているか確認
```
root@b0b404a14ad8:/workspace# ls nnUNet_results/Dataset502_ToothFairy3_consecutive/nnUNetTrainer__nnUNetPlans__2d/fold_0/
```
想定結果
```
checkpoint_best.pth debug.json training_log_2026_9_9_04_09_20.txt training_log_2026_9_9_04_21_18.txt
checkpoint_latest.pth progress.png training_log_2026_9_9_04_10_55.txt training_log_2026_9_9_05_10_08.txt
```
各ファイルの説明
- ceckpoint_latest.pth: 最後に保存された学習状態
- checkpoint_best.pth: これまでで性能が一番良かった時点のモデル
- progress.png: 学習曲線
- training_log*.txt: 学習ログ
- debug.json: 学習設定や環境情報
### 3D で学習したい
```
nnUNetv2_train 502 3d_fullres 0 -device cuda
```
### fold 数を増やしたい
```
nnUNetv2_train 502 2d 0 -device cuda
nnUNetv2_train 502 2d 1 -device cuda
nnUNetv2_train 502 2d 2 -device cuda
nnUNetv2_train 502 2d 3 -device cuda
nnUNetv2_train 502 2d 4 -device cuda
```
