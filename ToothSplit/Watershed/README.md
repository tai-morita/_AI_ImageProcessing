# Tooth Segmentation

## 概要
.vol データと 歯のラベルデータ(二値化)から歯の領域分割を行う。

## 環境
Python 3.13

## インストール
pip install -r requirements.txt

## 使用方法
- SplitTeethMain.py の SplitTeethMain() が実行関数。
- 引数で実行処理はしていないので、main 関数に直接入力する。
- SplitTeethMain(volume_file_path: str, teeth_binary_file_path: str)
  - input:
    - volume_file_path: volume データ
    - teeth_binary_file_path: volume_file_path の vol データから歯を抽出した 2 値化データ
  - output:
    - extracted_volume: volume_file_path から歯を抽出したデータ
    - edited_labeled_root_canal: 根管のラベルデータ
    - splitted_teeth_volume: 歯の領域分割データ