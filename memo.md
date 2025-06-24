# OpenCVやその他の知識を備忘録として入れる。
## 用語
- オープニング処理/クロージング処理(https://www.frontier.maxell.co.jp/blog/posts/22.html)
## OpenCV
- cv2.imread(filename, flags)  
  filenameを読み込んで、flagsで指定した方法で読み込む。
  - cv2.IMREAD_COLOR（デフォルト）：カラー画像で読み込み（アルファチャンネル無視）
  - cv2.IMREAD_GRAYSCALE：グレースケールで読み込み
  - cv2.IMREAD_UNCHANGED：アルファチャンネルも含めて読み込み
- cv2.cvtColor(src, code)  
  画像の色空間を変換する(カラー-> グレースケールなど)
  - src: 入力画像
  - code: 変換の種類
    - cv2.COLOR_BGR2GRAY: BGR-> グレースケール
    - cv2.COLOR_BGR2RGB : BGR-> RGB
- cv2.threshold(src, thresh, maxval, type)
  グレースケールに対して二値化を行う
  - src: 入力画像
  - thresh: 閾値
  - maxval: 最大値
  - type: 閾値処理タイプ
    - cv2.THRESH_BINARY: 閾値より大きい値をmaxvalに、それ以外は0にする
  - 出力はタプルで二つ[retval, dst]
    - retval: 実際に使用された閾値(cv2.THRESH_OTSUでは自動で閾値決定がされる)
    - dst   : 二値化後の画像(Numpy配列)
- cv2.merge([channel1, channel2, channel3])
  複数のチャンネルを１つのカラー画像に統合する。引数は各チャンネル画像(B, G, R)に対応する。
- cv2.morphorogyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
  バイナリ画像(閾値処理済み画像)から小さいノイズを除去鵜する
  - thresh: 入力画像
  - cv2.MORPH_OPEN: モルフォロジー処理の指定
    - cv2.MORPH_OPEN : オープニング処理(収縮(Erosion))-> 膨張(Dilation)
    - cv2.MORPH_CLOSE: クロージング処理(膨張-> 収縮)
  - kernel: カーネルの指定(3x3や5x5のnumpy配列を指定)
  - iterations=2: 繰り返し処理
- cv2.dilate(opening, kernel, iterations=3)
  モルフォロジー処理の膨張を使って、画像の背景領域を抽出する -> 膨張によりノイズをつぶす
- cv2.distanceTransform(opening, cv2.DIST_L2, MaskSize)
  距離返還を行う処理: 前景領域の中心を見つける。白い領域の各画素が、最も黒い画素までどれだけ離れているかを記録する
  - opening: 入力画像(白: 物体, 黒: 背景)
  - cv2.DIST_L2: 距離の計算方法
    - L2: ユークリッド距離
  - MaskSize: マスクサイズ。数値が大きいほど精度は上がる
- cv2.connectedComponents(binary_image)
  画像中の連結成分をラベリングする
  - binary_iname: 入力画像(二値化画像、背景が0、物体が255)
  - 出力: 連結成分の数、ラベリングされた画像