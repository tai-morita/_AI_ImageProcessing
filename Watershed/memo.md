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
- cv2.watershed(image, markers)
  watershedアルゴリズムより境界線を引く
  - image: 入力画像
  - markers: imageに対するマーカー画像
- cv2.Laplacian(img, cv2.CV_64F)
  ラプラシアンフィルタを用いて、エッジを検出する。さらに　cv2.CV_64Fによって64ビット浮動小数点にして出力される
  - ラプラシアンフィルタ -> 2階微分
  - img: ラプラシアンフィルタを適用する画像
  - cv2.CV_64F: 出力する形式の指定(cv2.CV_32Fなどもあり)
- cv2.convertScaleAbs(img)
  データを8ビットの正整数に変換する

## watershedアルゴリズムについて(概要)
watershedアルゴリズムは以下の手順で入力画像に境界を設ける
#### 必要なもの
- 入力画像: カラー画像であること
- マーカー画像: 背景、オブジェクト、未割当(0)(背景かオブジェクトかわからない場所)の3値で分けられている
#### アルゴリズム
1. エッジ強調を行い、勾配画像を作成する
2. 勾配の小さい画素から処理するため、勾配の小さい画素から順にキューに入れる(優先度付きキュー)
3. キュー内の画素がなくなるまで以下を繰り返す
  1. キューから画素pを取り出す
  2. pの近傍8(4)画素を調べる
    - 近傍の画素nが未割当(0)である場合、nにpのラベルを割り当て、nをキューに追加する
    - 近傍の画素nがpと異なる場合、nを境界部ラベル(-1)にする
    - 近傍の画素nがpと同じ、または境界部(-1)である場合、なにもしない

#### アルゴリズム(20260216)
これは **“マーカー付き・勾配画像”**に対する 優先度付きキュー方式の一例です。
plateau 公平性のため、**同じ値の束は FIFO で層処理（imd 併用）**します。 [watershed | PDF]


入力準備

画像 G（標高/勾配）。
前景マーカーと背景マーカーにユニークなラベルを付与して label を初期化。 [watershed | PDF]


優先度付きキュー（min-heap）を初期化

**“ラベル画素の未ラベル隣接”**だけを (priority=G値, 座標) で push。


処理ループ（PQ が空になるまで）

最小 priority の画素 p を pop。
p の 同値（同 priority）セットを一括収集し、FIFO キューに積んで plateau BFS を開始（このとき imd で距離/層を管理）。
plateau BFS 中：

隣接に 1つのラベルのみ → そのラベルで p を塗る。
隣接に複数（異なる）ラベル → 同距離層で衝突なら p を WSHED に。
未ラベル隣接は **同値優先度（同レベル）**で plateau 拡張、それ以外は その値を priority に PQ へ（“前線更新”）。


plateau BFS を抜けたら、**今回新たにラベルが付いた画素の“未ラベル隣接”**だけを PQ に追加。


終了

必要に応じて WSHED の事後処理（完全タイル化なら隣接ラベルで薄く埋める）を行う（厚い分水界を保持したい場合は埋めない）。 [watershed | PDF]




ポイント：全未分類画素を最初に PQ に入れない／plateau は同値束＋FIFO＋imd／境界は“異ラベルの同層衝突”でのみ確定。 [watershed | PDF]


参考になる具体的“失敗 → 修正”パターンまとめ


全画素を最初に PQ へ →
誤り：前線以外からも「水」が湧き、流域が破綻。
修正：前線のみ PQ へ（シード隣接→前線更新）。 [watershed | PDF]


plateau を単純な 2-1/2-2 判定だけで進める →
誤り：太い／曲がった分水界（Fig.10–11 型）。
修正：同値束＋imd＋FIFO 層処理で公平拡張（Fig.12/14 参照）。 [watershed | PDF]


「2つ以上で境界」を即決 →
誤り：不要に境界が太る／分岐。
修正：“同距離層での異ラベル衝突”のみ WSHED、それ以外は単一ラベルで塗る。 [watershed | PDF]


“存在しなければ再投入”だけで再PQ →
誤り：再投入スパム（無限・冗長）。
修正：未訪問かつ距離が伸びる場合のみ、または plateau BFS で一括処理。 [watershed | PDF]


背景を除外 →
誤り：背景側の谷が未定義→輪郭が閉じない／前景が膨張。
修正：背景マーカーを（1クラス）用意し、前線に含める。 [watershed | PDF]


4/8 近傍の効果を無視 →
誤り：望まぬ位相・連結性の分水界に。
修正：浸水近傍と分水界の連結性の関係を理解した上で選ぶ（4近傍→分水界8連結、8近傍→分水界4連結）。 [watershed | PDF]



必要であれば、**あなたの手順をベースにした“最小修正の擬似コード”**や、実データでの挙動比較スクリプト（FIFO版 vs PQ版、plateau例）も作成します。どのケース（医用、DEM、産業検査など）で使う予定かも教えてもらえれば、背景マーカーの作り方や勾配画像の設計（モルフォロジカルグラディエント／ガウシアン勾配など）の実務的なレシピも併せて最適化します。 [watershed | PDF]