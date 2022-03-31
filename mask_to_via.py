#! /usr/bin/env python3
#/-*- encoding: utf-8 -*- /

# mask2via_v4.py
# アノテーション部分が白抜きされた背景黒色のマスク画像群と元画像（画像サイズ
# 取得用）から、VIA JSONデータを作成する
# ファイルは同名（ただし元画像は.JPG、マスクは.png）である必要あり
#
# 2020-05 ShapeMask向けに作成、マスクはバイナリ画像を想定
# 2020-05-27 マスク画像がバイナリ以外のグレースケールの場合に対応
#            （クラス分けは非対応）
# 2020-05-28 [v2] 1マスク1枚（1つの元画像に複数のマスク画像が紐づく）の場合に
#                 対応
# 2020-05-28 [v3] 大量の画像を処理する場合用に並列化、1秒500枚くらい
# 2020-06-22 [v4] ポリゴンの穴を埋める処理を追加
# 2020-10-29      threshold が無視されるバグを修正
# 2021-01-14 [v5] IoU 計算に失敗することがある問題を修正
# 2021-01-20      多クラスのマスクに対応、クラス名でフィルタする機能を追加


from pathlib import Path
import re
import json
import argparse
import logging
import multiprocessing
import itertools

import numpy as np
from PIL import Image, ExifTags
from tqdm.contrib.concurrent import process_map
from shapely.geometry import Polygon
import shapely.errors
import cv2
import skimage.measure
import skimage.io

logging.basicConfig(level=logging.WARNING)


def reorient(array, im_path):
    ''' im で指定された画像のEXIFメタデータを参照して、
    画像の回転を打ち消す方向に Numpy 配列を回転します
    引数: im_path [Path] EXIFデータを参照する画像
          array [np.ndarray] 回転する画像
    返り値: rot_array [np.ndarray] EXIFデータの指示と逆方向に回転した配列
    '''
    im = Image.open(im_path)

    # https://stackoverflow.com/a/26928142/13191651
    try:
        for orientation in ExifTags.TAGS:
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(im._getexif().items())

        if exif[orientation] == 3:
            rot_array = np.rot90(array, 2)
        elif exif[orientation] == 6:
            rot_array = np.rot90(array, 3)
        elif exif[orientation] == 8:
            rot_array = np.rot90(array, 1)
        else:
            rot_array = array

        return rot_array

    except (AttributeError, KeyError, IndexError):
        logging.info('Exif metadata missing. Orientation correction skipped.')
        return array


def polygonize(segm):
    '''マスクからポリゴンを生成します
    引数: segm [np.array] (H, W)、値域0-1の配列
    返り値: xs [list] 生成したポリゴンのX座標
                      xs[i] は i 番目のポリゴン
            ys [list] 生成したポリゴンのY座標
                      ys[i] は i 番目のポリゴン
    '''
    segm = (segm * 255).astype(np.uint8)
    layers = np.unique(segm)    # ユニークなピクセル値 バイナリ画像なら[0,255]
    logging.debug(layers)

    xs = []
    ys = []

    # ピクセル値ごとにポリゴン生成ロジックを実行
    for i in layers:
        if i == 0:
            continue    # 黒（明度0）は背景なのでスキップ
        layer = (segm == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            for contour in contours:
                contour = contour.reshape((-1, 2))
                contour_smooth = skimage.measure.approximate_polygon(
                    contour, tolerance=0.95)  #[[x0, y0], [x1, y1], ...]
                all_points_x, all_points_y = contour_smooth.T.tolist()
                xs.append(all_points_x)
                ys.append(all_points_y)

    assert len(xs) == len(ys), "Number of polygons does not match. Odd."
    return xs, ys


def resize(xi_yi, source_size, target_size):
    '''与えられたポリゴンを、別の画像の大きさにリサイズします
    引数: xi_yi [list] ポリゴンのリスト
                       xi_yi[0] はx軸のポリゴン集合、[1] はy軸のポリゴン集合
          source_size [tuple] リサイズ元の画像サイズ (W, H)
          target_size [tuple] リサイズ先の画像サイズ (W, H)
    返り値: (list, list) リサイズ後のポリゴンの (x座標, y座標)
    '''
    xi, yi = xi_yi
    source_width, source_height = source_size
    target_width, target_height = target_size
    xi = [
        [int(x * target_width / source_width) for x in xj] for xj in xi]
    yi = [
        [int(y * target_height / source_height) for y in yj] for yj in yi]

    return xi, yi


def cleanse_polygons(xs, ys, th=0.1):
    '''ポリゴンをきれいにします
    引数: xs [list] ポリゴンのx座標一覧、all_points_x のリスト
          ys [list] ポリゴンのy座標一覧、all_points_y のリスト
          th [float] ポリゴンをマージするしきい値、(0.0, 1.0)
    返り値: xs_ret [list], ys_ret[list]
                引数と同じフォーマット、穴あき除去済み
    TODO: 内包判定とマージを分離
          shapelyオブジェクトを受け渡すように変更（shapely→[xs, ys]変換は分離）
    '''

    polygons = [
        list(zip(xs[i], ys[i])) for i in range(len(xs)) if len(xs[i]) > 3]
        # i は [0,ポリゴンの数-1]
    polygon_objects = [Polygon(coords) for coords in polygons]

    holes = []
    union = []
    union_partial = []
    for p, q in itertools.combinations(polygon_objects, 2):
        # いずれかのポリゴンが内包するポリゴンは削除
        if p.contains(q):
            holes.append(q)

        # IoU >= th のオブジェクトは和集合をとり、もとのオブジェクトは削除
        # v5: なぜか p.union(q).area == 0 の場合があるので回避
        if (p.union(q).area > 0) \
                and (p.intersection(q).area / p.union(q).area >= th):
            union.append(p.union(q))
            union_partial += [p, q]

    filt_polygon_objects = [
        i for i in polygon_objects if i not in (holes + union_partial)] + union

    filt_polygons = [
        list(p.exterior.coords) for p in filt_polygon_objects
        if isinstance(p, Polygon)]   # PolygonをXY座標ペアに
    xs = [[int(coord[0]) for coord in polygon] for polygon in filt_polygons]
    ys = [[int(coord[1]) for coord in polygon] for polygon in filt_polygons]

    return xs, ys

def get_label(name):
    '''与えられたマスク画像名をパースし、ラベル名を返します
    '''
    return re.search(r'(?<=label)[^\.]+', name).group(0)

def get_via(image_path, mask_paths, do_reorient):
    '''
    image_path の画像に対するVIAアノテーションを自動生成します
    引数: image_path [Path] 画像ファイルパス
          masks [list] マスク画像のパス [Path] のリスト
          do_reorient [bool] image_path の画像のEXIFメタデータに従いアノテー
                             ションを回転させるかどうか
    返り値: via_key [str] VIA JSONのエントリのキー
            via_val [dict] VIA JSONのエントリの値
    '''
    ## オリジナル画像の幅と高さを取得する
    ## TODO: 遅いのでスキップ可能にする
    logging.debug(f"Generating via for {image_path}...")
    original_im = Image.open(image_path)

    ## 二値マスクをポリゴン化
    xs = []
    ys = []
    labels = []
    for mask_path in mask_paths:
        label = get_label(mask_path.name)
        mask = skimage.io.imread(
            str(mask_path), as_gray=True, plugin='matplotlib')
        if do_reorient:
            mask = reorient(mask, image_path)
        xi, yi = polygonize(mask)
        xi, yi = resize((xi, yi), mask.shape[:2], original_im.size)

        xs += xi
        ys += yi
        labels += [label] * len(xs)

    ## ポリゴンの内包、重複を検査
    try:
        xs, ys = cleanse_polygons(xs, ys)
    except shapely.errors.TopologicalError:
        # Polygon の輪郭に「結び目」（2辺が交差する部分）がある場合はエラー
        logging.warning(
            f"画像 {image_path.name} の内包判定に失敗しました。"
            "ポリゴンは削除されます")
        xs, ys = ([], [])

    ## VIA JSON のエントリを作成
    regions = []
    for i, _ in enumerate(xs):
        all_points_x = xs[i]
        all_points_y = ys[i]
        label_name = labels[i]
        if len(all_points_x) <= 5 or len(all_points_x) <= 10:
            pass
        else:
            region = {
                'shape_attributes': {
                    'name': 'polygon',
                    'all_points_x': all_points_x,
                    'all_points_y': all_points_y
                },
                'region_attributes' : {
                    'name': label_name
                }
            }
            regions.append(region)

    filesize = image_path.stat().st_size
    via_key = f'{image_path.name}{filesize}'
    via_val = {
        'filename': image_path.name,
        'size': filesize,
        'regions': regions,
        'file_attributes': {}
    }

    return via_key, via_val


def min_conf(name, th):
    '''次のようなファイル名 "righttop_00_conf0.98.png" を解析し、
    confに続いて指定された確信度が th 以上か判定します
    引数: name [str] ファイル名
          th [float] 最低確信度
    返り値: [bool] 確信度がしきい値以上ならTrue、さもなくばFalse
    '''
    m = re.search(r'(?<=conf)[01]\.\d{2}', name)

    if m is None:
        return False

    conf = float(m.group(0))
    return conf >= th


def in_label(name, label):
    '''次のようなファイル名 "righttop_00_conf0.98_labelleaf_1.png" を解析し、
    labelに続いて指定されたラベルに引数 label で指定した文字列が含まれているか
    判定します
    引数: name [str] ファイル名
          label [str] 検索条件
    返り値: [bool] 指定した文字がラベルに含まれるならTrue、さもなくばFalse
    '''
    m = re.search(r'(?<=label)[^\.]+', name)  # label から 拡張子の直前まで検索
    if m is None:
        return False
    this_label = m.group(0)

    return label in this_label


class _map_fn():
    '''並列処理のさい実行する関数
    共通の引数を保持させるため（だけ）にクラス化
    '''
    def __init__(self, mask_dir, do_reorient, threshold, label):
        self.mask_dir = mask_dir
        self.do_reorient = do_reorient 

        # しきい値、ラベル名の条件に合致するマスク画像を取得
        check_conf_and_label = \
            lambda fname: min_conf(fname, threshold) \
                and (in_label(fname, label) if label is not None else True)
        self.all_masks = [
            mask for mask in self.mask_dir.glob('*.png')
            if check_conf_and_label(mask.name)]


    def __call__(self, eval_image):
        '''元画像とマスク画像を読み込んでVIAのエントリをつくります
        引数: eval_image [Path] 元画像
        返り値: k [str] VIAのキー
                v [dict] VIAの値
        '''
        masks = [
            mask for mask in self.all_masks \
            if mask.name.startswith(eval_image.stem)]
        logging.debug(f'Filtered masks: {masks}')

        if len(masks) == 0:
            logging.info(f'No mask found for {eval_image.name}. Skipping.')
            return None

        k, v = get_via(eval_image, masks, self.do_reorient)
        return k, v


def main(args):
    image_dir = Path(args.images)
    mask_dir = Path(args.masks)
    export_to = args.export_to
    assert 0 < args.threshold and args.threshold < 1, \
        '最低信頼度は0と1のあいだに設定してください'

    ## マスク作成 並列化対応済み
    map_fn = _map_fn(mask_dir, args.do_reorient, args.threshold, args.label)
    eval_images = [
        f for f in image_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png')]
    workers = multiprocessing.cpu_count()
    map_ret = process_map(
        map_fn, eval_images, max_workers=workers, ascii=True, chunksize=1)
    map_ret = [i for i in map_ret if i is not None]
    eval_result = dict(map_ret)   # 書き出される辞書データ

    ## 書き出し
    with open(export_to, mode='w') as f:
        json.dump(eval_result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='アノテーションが白抜きされた背景黒色のバイナリ画像群か' \
        'らVIA JSONデータを作成する')
    parser.add_argument(
        "-i", "--images", required=True, metavar="/read/images/in/this/dir/",
        help="アノテーション対象画像のディレクトリパス")
    parser.add_argument(
        "-m", "--masks", required=True, metavar="/masks/are/in/this/dir/",
        help="マスク画像のパス")
    parser.add_argument(
        "-j", "--export_to", required=True,
        metavar="/export/to/that/via_region_data.json",
        help="VIA JSON の書き出し先")
    parser.add_argument(
        "-r", "--do_reorient", default=False, type=bool,
        help="EXIFメタデータを読み込んでアノテーションを回転させるかどうか。")
    parser.add_argument(
        "-th", "--threshold", default=0.5, type=float,
        help="検出時最低信頼度（デフォルト:0.5）。")
    parser.add_argument(
        "-l", "--label", default=None, type=str,
        help="ラベル。指定した文字列を含むクラスのポリゴンのみ書き出す。" \
        "指定しなければクラスでフィルタしない")
    args = parser.parse_args()

    main(args)
