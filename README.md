# ShapeMask

## 概要

このレポジトリは、AI病虫害画像診断システム　前景抽出モデル（ShapeMask[^1]）の学習・推論プログラムを格納しています。

なお、本プログラムは、[tensorflow/tpu](https://github.com/tensorflow/tpu/tree/master/models/official/detection) の一部を改変したものです。

## 動作環境

本プログラムの動作環境は下記のとおりです。

* Ubuntu 16.04 Xenial

学習・評価には、深層学習用の高速演算装置 [TPU](https://cloud.google.com/tpu/) を使うことができます。TPU を使用するには、[Google Cloud Platform](https://cloud.google.com/) のアカウントが必要です。

## インストール手順

環境構築には、[Docker Engine](https://docs.docker.com/engine/) を使用します。GPU で推論する場合は、NVIDIA ドライバと [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) をインストールしてください。

動作環境（Docker コンテナ）は、下記のコマンドで作成できます。

```console
$ cd /path/to/cloned/repo   # レポジトリへのパス
$ docker build -t shapemask ./
$ docker run -itdv /path/to/cloned/repo:/work/repo --name shapemask shapemask bash
```

また、本プログラムは [Google Colaboratory](https://colab.research.google.com/) 上でも動作します。 Colaboratory での実行方法は、[付属のノートブック](shapemask.ipynb) をご参照ください。

## データセットの準備

ShapeMask モデルを構築するには、物体（葉や果実など）をポリゴン形状でアノテーションしたデータが必要です。

### 画像データの準備

AI病虫害画像診断システムを構築するには、病害虫画像のデータセットが必要です。
構築に使用した画像とメタデータは、病害虫被害画像データベース
(https://www.naro.affrc.go.jp/org/niaes/damage) で公開しています。

学習・評価に使う画像を、下記のようなディレクトリ構造で配置します。
```
my_dataset/
    train/
        100000_20201231123456_01.JPG
        200000_20201223234501_01.JPG
        ...
    val/
        300000_20201231123456_01.JPG
        400000_20201223234501_01.JPG
        ...

{train, validation}: train は学習用、validation は評価用のサブセットです。
```

元画像のサイズ（3000x4000など）のままだと、学習時間が非常に長くなります。[ImageMagick](https://imagemagick.org/) などを使用して画像をリサイズしてください。

```console
$ find /path/to/my_dataset/ -iname '*.JPG' \
    | xargs -n16 -P4 mogrify -auto-orient -resize 256x256\!
```

### アノテーションデータの準備

アノテーションデータは、[VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) (VIA) で作成することができます。下記の手順で、アノテーションデータを作成します。

1. Web ブラウザで VIA を開きます。
1. 左ペインの「Add Files」で、画像を開きます。複数の画像を同時に開くこともできます。
1. 左ペインの「Region Shape」で、ポリゴン（五角形のアイコン）を選択します。
1. 画像上をクリックして、ポリゴン形状のアノテーションを作成します。クリックするたびに、ポリゴンの頂点が追加されます。ポリゴンを閉じるには、ダブルクリックします。
1. ポリゴンのメタデータを追加します。
    1. 左ペインの「Attributes」から「Region Attributes」を選択します。
    1. テキストボックスに「name」と入力して「+」ボタンを押します。
    1. 「Type」属性を「text」に設定します。
    1. 「Toggle Annotation Editor」をクリックして、アノテーションエディタを開きます。
    1. 「Region Annotations」タブを選択し、「name」属性のテキストボックスに物体の種類（"leaf" や "fruit" など）を入力します。
1. すべての画像に対して、上記の手順を繰り返します。
1. アノテーションの作成が終わったら、「Annotation」メニューの「Export Annotations (as json)」をクリックして、アノテーションデータをダウンロードします。

### MS-COCO 形式への変換

TFRecord に変換するには、まず VIA 形式のアノテーションデータを MS-COCO 形式に変換する必要があります。
MS-COCO 形式のアノテーションデータを生成するには、変換ツール `via2coco.py` を使います。

```console
$ ## 学習用データの変換
$ docker exec -it shapemask python /work/repo/tools/via2coco.py \
    --inputJson /path/to/my_dataset/train/via_region_data.json \
    --imagesDir /path/to/my_dataset/train/ \
    --outputPath /path/to/my_dataset/coco/ \
    --dataType train \
    --segdata 1

$ ## 評価用データの変換
$ docker exec -it shapemask python /work/repo/tools/via2coco.py \
    --inputJson /path/to/my_dataset/val/via_region_data.json \
    --imagesDir /path/to/my_dataset/val/ \
    --outputPath /path/to/my_dataset/coco/ \
    --dataType val \
    --segdata 1
```

### TFRecord 形式への変換

学習前に、データセットを TFRecord 形式に変換（シリアル化）する必要があります。シリアル化には `create_coco_tf_record.py` を使います。

次のコマンドで、TFRecord に変換します。

```console
$ COCO_ROOT="/path/to/my_dataset/coco/"
$ TFR_ROOT="/path/to/my_dataset/coco_tfrecord/"

$ docker exec -it shapemask python /work/repo/tools/create_coco_tf_record.py \
    --logtostderr \
    --include_masks \
    --image_dir="${COCO_ROOT}/images/train2017/" \
    --object_annotations_file="${COCO_ROOT}/annotations/instances_train2017.json" \
    --caption_annotations_file="${COCO_ROOT}/annotations/captions_train2017.json" \
    --output_file_prefix="${TFR_ROOT}/train" \
    --num_shards=4
$ docker exec -it shapemask python /work/repo/tools/create_coco_tf_record.py \
    --logtostderr \
    --include_masks \
    --image_dir="${COCO_ROOT}/images/val2017/" \
    --object_annotations_file="${COCO_ROOT}/annotations/instances_val2017.json" \
    --caption_annotations_file="${COCO_ROOT}/annotations/captions_val2017.json" \
    --output_file_	prefix="${TFR_ROOT}/val" \
    --num_shards=1

$ cp "${COCO_ROOT}/annotations/instances_val2017.json" "${OUTPUT_DIR}/"
```

### アップロード（TPU 使用時のみ）

TPU で学習・評価する場合、データセットを Google Cloud Storage にアップロードする必要があります（Cloud Storage 以外の場所に置いたデータセットをTPUで読み込むことはできません）。

[`gsutil`](https://cloud.google.com/storage/docs/gsutil) コマンドを使って、TFRecord 形式のデータセットを Google Cloud Storage にアップロードします。

```console
$ gsutil -m cp -r /path/to/my_dataset/coco_tfrecord gs://my-bucket/my_dataset/
```

## 学習方法

### 設定ファイルの作成

ShapeMask の学習時設定（データセットのファイルパス、学習ステップ数など）は、YAML 形式のファイルに記述します。
設定ファイルサンプル（[config_sample.yaml](config_sample.yaml)）を参考にして、設定してください。

主な設定項目は下記のとおりです。

* `num_classes`: データセットに含まれるクラス（物体の種類）の数。背景（BG）を含めて数えるので、2以上。
* `iterations_per_loop`: 学習途中のモデル（チェックポイント）を書き出すステップ間隔。
* `train_batch_size`: 学習時のバッチサイズ（画像サイズ256px の場合、TPU v2-8 では 256 程度）
* `total_steps`: 学習ステップ数。エポック数を決めて学習する場合、下記の式でステップ数を求められる。
`(ステップ数) = (画像枚数) * (エポック数) / (バッチサイズ)`
* `learning_rate`: 学習率。
* `checkpoint`: 事前学習モデル。指定したモデルから読み込んだパラメータを、ネットワークの初期値として学習する。
* `val_json_file`: 評価用データのアノテーション（`instances_val2017.json`）のファイルパス。
* `eval_file_pattern`: 評価用データ（TFRecord）のファイルパス（glob 形式で指定）。
* `eval_samples`: 評価用データの画像数
* `aug_rand_rot`: 180度の回転水増しを適用するかどうか（bool）。
* `aug_rand_rot_half_pi`: 90度の回転水増しを適用するかどうか（bool）。
* `output_size`: 推論結果の画像サイズ（px）。

### 学習コマンドの実行

学習は、下記のコマンドで実行することができます。

```console
$ docker exec -it shapemask python /work/repo/main.py \
    --model="shapemask" \
    --use_tpu="True" \
    --tpu="grpc://0.0.0.0:8470" \
    --num_cores="8" \
    --model_dir="gs://path/to/model/" \
    --mode="train" \
    --config_file="gs://path/to/config.yaml" \
```

各コマンドライン引数の説明は、下記のとおりです。

* `--use_tpu`: TPU を使用するか（bool、False ならば CPU/GPU を使用）
* `--tpu`: TPU アドレス（`grpc://[TPU ノードのGRPCアドレス]:8470`、CPU/GPU 学習時は不要）
* `--num_cores`: コア数（例: 1枚のGPU -> 1、TPU v2-8 -> 8）
* `--model_dir`: 学習済みモデルの保存先ディレクトリ
* `--mode`: 動作モード (学習時は `train`)
* `--config_file`: 設定ファイルのパス

学習が完了すると、`model_dir` に指定したディレクトリに学習済みモデルが保存されます。

## 評価方法

評価は、下記のコマンドで実行することができます。

```console
$ docker exec -it shapemask python /work/repo/main.py \
    --model=shapemask \
    --use_tpu="True" \
    --tpu="grpc://0.0.0.0:8470" \
    --num_cores="8" \
    --model_dir="gs://path/to/model/" \
    --mode="eval" \
    --config_file="gs://path/to/config.yaml" \
    --params_override='{eval:{eval_batch_size: 1}}'
```

各コマンドライン引数の説明は、下記のとおりです。

* `--use_tpu`: TPU を使用するか（bool、False ならば CPU/GPU を使用）
* `--tpu`: TPU アドレス（`grpc://[TPU ノードのGRPCアドレス]:8470`、CPU/GPU 学習時は不要）
* `--num_cores`: コア数（例: 1枚のGPUで推論 → 1、TPU v2-8で推論 → 8）
* `--model_dir`: 学習済みモデルの保存先ディレクトリ
* `--mode`: 動作モード (学習時は `train`)
* `--config_file`: 設定ファイルのパス

評価が完了すると、評価結果（IoUなど）が標準出力に書き出されます。

## 推論方法

未知の画像に対する推論は、下記のコマンドで実行することができます。

```console
$ docker exec -it shapemask python /work/repo/inference_via.py \
    --image_size="256" \
    --model=shapemask \
    --checkpoint_path="/path/to/model.ckpt-10000" \
    --config_file="/path/to/config.yaml" \
    --image_file_pattern="/path/to/image/dir/*.JPG" \
    --label_map_file="/path/to/label_map.csv" \
    --export_to="/path/to/export/masks/to/"
```

各コマンドライン引数の説明は、下記のとおりです。

* `--image_size`: 画像サイズ
* `--model`: モデル名（`shapemask` を指定）
* `--checkpoint_path`: 学習したモデルのチェックポイントファイル
* `--config_file`: 設定ファイルのパス
* `--image_file_pattern`: 画像ファイルのパターン（glob形式で指定）
* `--label_map_file`: クラスのインデックスとクラス名の定義ファイル（CSV形式）
* `--export_to`: 推論結果マスクの書き出し先

推論結果は、`--export_to` で指定したディレクトリ内にバイナリマスク画像として書き出されます。

さらに下記のコマンドを実行すると、書き出したマスク画像から VIA 形式のアノテーションファイルを作成することができます。

```console
$ docker exec -it shapemask python /work/repo/mask_to_via.py \
    --images "/path/to/images/" \
    --masks "/path/to/masks/" \
    --export_to "/path/to/via_region_data.json"
```

各コマンドライン引数の説明は、下記のとおりです。

* `--images`: 元画像のあるディレクトリパス。
マスク画像から生成したポリゴンは、`--images` で指定したディレクトリにある画像の大きさに引き延ばされる。
画像をリサイズして推論した場合、リサイズ前の画像ディレクトリを指定することで、もとの画像サイズでアノテーションを生成できる。
* `--masks`: マスク画像のあるディレクトリパス。
* `--export_to`: アノテーションファイルの書き出し先

## ライセンス

本プログラムは、Apache License 2.0 で提供されます。詳細は、[LICENSE](LICENSE) ファイルをご参照ください。

## 改変箇所

オリジナルのプログラム (https://github.com/tensorflow/tpu/tree/master/models/official/detection) との主要な差異は、下記のとおりです。

* 学習時に、チェックポイントの最大保存数を指定するオプションを追加（`--keep_checkpoint_max`）。
* 評価時に、すべてのチェックポイントを順次評価する機能を追加（`--eval_all_ckpts`）
* 推論結果を、VIA JSON 形式のアノテーションデータとして書き出す機能を追加（`inference_via.py`, `mask_to_via.py`）
 
[^1]: https://arxiv.org/abs/1904.03239
