# -*- coding: utf-8 -*
#
# via2coco_2017.py

import json
import logging
from pathlib import Path
import copy
import multiprocessing
import argparse

import numpy as np
import skimage.draw
import skimage.io
import cv2

parser = argparse.ArgumentParser(description='MyScript')
parser.add_argument('--inputJson', type=Path, default="./viajson.json")
parser.add_argument('--imagesDir', type=Path, default="./Image")
parser.add_argument('--classJson', default=None)
parser.add_argument('--outputPath', type=Path, default="./Data")
parser.add_argument('--dataType', default="train")
parser.add_argument('--rect2seg', type=int, default=0)
parser.add_argument('--segdata', type=int, default=0)
parser.add_argument('--rotate', type=int, default=0)
args = parser.parse_args()

infoArr = {
    "description": "COCO 2017 Dataset",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2017,
    "contributor": "COCO Consortium",
    "date_created": "2017/09/01"
}
infoArr_pn = {
    "description": "COCO 2018 Panoptic Dataset",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2018,
    "contributor": "https://arxiv.org/abs/1801.00868",
    "date_created": "2018-06-01 00:00:00.0"
}
licensesArr = [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    },
]

args.outputPath.mkdir(exist_ok=True, parents=True)

annotations_dir = args.outputPath/'annotations'
annotations_dir.mkdir(exist_ok=True, parents=True)

out_image_dir = args.outputPath/"images"/f"{args.dataType}2017"
out_image_dir.mkdir(exist_ok=True, parents=True)

out_pano_dir = annotations_dir/f"panoptic_{args.dataType}2017"
out_pano_dir.mkdir(exist_ok=True, parents=True)

if args.segdata > 0:
    out_semantic_dir = \
        annotations_dir/f"panoptic_{args.dataType}2017_semantic_trainid_stff"
    out_semantic_dir.mkdir(exist_ok=True, parents=True)


CLASS_COLOR_LIST = {
       1 : [20,  20,  20,  1],
       2 : [40,  40,  40,  1],
       3 : [60,  60,  60,  1],
       4 : [80,  80,  80,  1],
       5 : [100, 100, 100, 1],
       6 : [120, 120, 120, 1],
       7 : [140, 140, 140, 1],
       8 : [160, 160, 160, 1],
       9 : [180, 180, 180, 1],
      10 : [200, 200, 200, 1],
      11 : [220, 220, 220, 1],
      12 : [230, 230, 230, 1],
      13 : [250, 250, 250, 1],
    }

CLASS_COLOR_LIST_SEG = {
       1  : [0, 0, 0, 1],
       2  : [1, 1, 1, 1],
       3  : [2, 2, 2, 1],
       4  : [3, 3, 3, 1],
       5  : [4, 4, 4, 1],
       6  : [5, 5, 5, 1],
       7  : [6, 6, 6, 1],
       8  : [7, 7, 7, 1],
       9  : [8, 8, 3, 1],
       10 : [9, 9, 9, 1],
       11 : [10, 10, 10, 1],
       12 : [11, 11, 11, 1],
       13 : [12, 12, 12, 1]
    }

ROTATE_ARR = [0]
if args.rotate > 0:
    ROTATE_ARR = [0, 90, 270]

with args.inputJson.open("r") as via_fp:
    via_json = json.load(via_fp)

classArr = [
        {
            "supercategory": "supercat",
            "isthing": 1,
            "id": 1,
            "name": "leaf"
        },
        {
            "supercategory": "supercat",
            "isthing": 1,
            "id": 2,
            "name": "fruit"
        },
    ]

if args.classJson is not None:
    with args.classJson.open('r') as class_fp:
        class_json = json.load(class_fp)

    classArr = class_json['categories']


def flatten(l):
    '''Flattens a list.
    '''
    return [i for sublist in l for i in sublist]


def area(x,y):
    '''Calculates the area (px^2) enclosed by the given x and y coordinates.
    Original code by Mahdi: https://stackoverflow.com/a/30408825/13191651
    '''
    return 0.5 * np.abs(
            np.dot(x, np.roll(y,1)) \
            - np.dot(y,np.roll(x,1))
        )

def generate_coco_files(input_data):
    val, imgId, ann_id_arr, rotate_val = input_data
    filename = val['filename']
    regions = val['regions']
    logging.debug(f'Selecting {filename}')

    re_annotationArr = []
    re_annotationArr_pn = []

    input_image_tmp = skimage.io.imread(
        args.imagesDir/filename, plugin='matplotlib')
    logging.debug(f'shape(input_image_tmp): {input_image_tmp.shape}')
    image_width_tmp = input_image_tmp.shape[1]
    image_height_tmp = input_image_tmp.shape[0]

    if rotate_val > 0:
        input_image = np.rot90(input_image_tmp, -1*int(rotate_val/90))
    else:
        input_image = input_image_tmp

    logging.debug(f'shape(input_image): {input_image.shape}')

    image_width = input_image.shape[1]
    image_height = input_image.shape[0]

    out_filename = "{:012}.png".format(imgId)

    skimage.io.imsave(
        out_image_dir/out_filename, input_image, check_contrast=False)

    label_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    if args.segdata > 0:
        # ポリゴン領域だけを黒塗りにしたPNG画像を作成する
        seg_image = np.full(
            (image_height, image_width, 1), 255, dtype=np.uint8)

    tmp_anotationArr_pn = {
        "segments_info": [],
        "file_name":out_filename,
        "image_id":imgId
    }

    ann_id_arr_cnt = 0
    for region in regions:
        className = ""
        if "name" in region['region_attributes']:
            className = region['region_attributes']['name']
        elif "class" in region['region_attributes']:
            className = region['region_attributes']['class']
        classIds = [x['id'] for x in classArr if x['name'] == className]
        classId = classIds[0] if len(classIds) else ''

        if classId == '':
            print(f"{filename} was skipped. (Class not found in hardcoded dir)")
            continue

        shape_attr = region['shape_attributes']

        if shape_attr['name'] == "polygon" and (
                len(shape_attr['all_points_x']) < 3 \
                or len(shape_attr['all_points_y']) < 3):
            continue

        anoId = ann_id_arr[ann_id_arr_cnt]
        class_color = CLASS_COLOR_LIST[int(classId)]
        seg_color = CLASS_COLOR_LIST_SEG[int(classId)]

        tmp_anotationArr = {
            "id": anoId,
            "image_id": imgId,
            "segmentation": [],
            "area": 0.0,
            "iscrowd": 0,
            "category_id": int(classId)}
        tmp_anotationArr_pn_sub = {
            "id": anoId,
            "category_id": int(classId),
            "iscrowd": 0,
            "bbox": [],
            "area": 0.0}

        if shape_attr['name'] == 'polygon' and args.rect2seg == 0:
            if len(shape_attr['all_points_x']) < 3 \
               or len(shape_attr['all_points_y']) < 3:
                continue

            points_x_tmp = np.array(shape_attr['all_points_x'])
            points_y_tmp = np.array(shape_attr['all_points_y'])

            tmp_range = range(points_x_tmp.shape[0])
            points_xy_tmp = np.insert(
                points_y_tmp, tmp_range, points_x_tmp[tmp_range])

            mat = cv2.getRotationMatrix2D(
                (image_width_tmp / 2, image_height_tmp / 2), -1*rotate_val, 1.0)
            points_xy_tmp2 = cv2.transform(points_xy_tmp.reshape([-1,1,2]), mat)
            points_x = points_xy_tmp2[:,:,0].reshape(-1)
            points_y = points_xy_tmp2[:,:,1].reshape(-1)
            points_x = points_x + int((image_width - image_width_tmp)/2)
            points_y = points_y + int((image_height - image_height_tmp)/2)
            points_x = points_x.tolist()
            points_y = points_y.tolist()

            #print("[ "+str(imgId)+" ] points_x_tmp : "+str(points_x_tmp))
            #print("[ "+str(imgId)+" ] points_y_tmp : "+str(points_y_tmp))
            #print("[ "+str(imgId)+" ] points_x : "+str(points_x))
            #print("[ "+str(imgId)+" ] points_y : "+str(points_y))

            xmax = max(points_x)
            xmin = min(points_x)
            ymax = max(points_y)
            ymin = min(points_y)

            if xmax < 0:
                xmax = 0
            elif xmax >= image_width:
                xmax = image_width
            if xmin < 0:
                xmin = 0
            elif xmin >= image_width:
                xmin = image_width

            if ymax < 0:
                ymax = 0
            elif ymax >= image_height:
                ymax = image_height-1
            if ymin < 0:
                ymin = 0
            elif ymin >= image_height:
                ymin = image_height-1

            tmp_anotationArr["bbox"] = [xmin, ymin, xmax-xmin, ymax-ymin]
            tmp_anotationArr_pn_sub["bbox"] = [xmin, ymin, xmax-xmin, ymax-ymin]
            tmp_xyArr = []
            points_x_tmp = []
            points_y_tmp = []

            for xpix, ypix in zip(points_x, points_y):
                if xpix < 0:
                    tmp_xyArr.append(0)
                elif xpix >= image_width:
                    tmp_xyArr.append(image_width-1)
                    points_x_tmp.append(image_width-1)
                else:
                    tmp_xyArr.append(xpix)
                    points_x_tmp.append(xpix)

                if ypix < 0:
                    tmp_xyArr.append(0)
                elif ypix >= image_height:
                    tmp_xyArr.append(image_height-1)
                    points_y_tmp.append(image_height-1)
                else:
                    tmp_xyArr.append(ypix)
                    points_y_tmp.append(ypix)

                #tmp_xyArr.append(xpix)
                #tmp_xyArr.append(ypix)

            area_px2 = int(area(points_x, points_y))
            tmp_anotationArr["area"] = area_px2
            tmp_anotationArr_pn_sub["area"] = area_px2

            rr, cc = skimage.draw.polygon(points_y_tmp, points_x_tmp)
            label_image[rr, cc, 0] = class_color[0]
            label_image[rr, cc, 1] = class_color[1]
            label_image[rr, cc, 2] = class_color[2]

            if args.segdata > 0:
                seg_image[rr, cc] = seg_color[0]

            tmp_anotationArr["segmentation"].append(tmp_xyArr)

        elif shape_attr['name'] == 'rect' and args.rect2seg == 1:
            xmax = shape_attr['x'] + shape_attr['width']
            xmin = shape_attr['x']
            ymax = shape_attr['y'] + shape_attr['height']
            ymin = shape_attr['y']

            tmp_anotationArr["area"] = \
                shape_attr['width'] * shape_attr['height']
            tmp_anotationArr["bbox"] = [
                shape_attr['x'], shape_attr['y'],
                shape_attr['width'], shape_attr['height']]
            tmp_anotationArr_pn_sub["bbox"] = [
                shape_attr['x'], shape_attr['y'],
                shape_attr['width'], shape_attr['height']]
            tmp_xyArr = [
                xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin]

            tmp_anotationArr["segmentation"].append(tmp_xyArr)
            print(tmp_anotationArr)

        if len(tmp_anotationArr["segmentation"]) > 0:
            re_annotationArr.append(tmp_anotationArr)
            tmp_anotationArr_pn["segments_info"].append(tmp_anotationArr_pn_sub)
            ann_id_arr_cnt += 1

    if len(tmp_anotationArr_pn["segments_info"]) > 0:
        re_annotationArr_pn.append(tmp_anotationArr_pn)
        skimage.io.imsave(
            out_pano_dir/out_filename, label_image, check_contrast=False)
        if args.segdata > 0:
            skimage.io.imsave(
                out_semantic_dir/out_filename, seg_image, check_contrast=False)

    print(f"{imgId}: {filename} was written.")

    image_meta = {
        "license": 1,
        "flickr_url": '',
        "coco_url": '',
        "date_captured": "2000-01-01 12:34:56",
        'file_name': out_filename,
        'width': input_image.shape[1],
        'height': input_image.shape[0],
        'id': imgId
    }

    return [image_meta, re_annotationArr, re_annotationArr_pn]

## 並列実行用に、generate_coco_files() に入力するデータを作成
## {VIA のポリゴン（region）} * {rotate_val} 個のデータを作成する
imgId = 0
anoId = 0
via_json_keys = sorted(via_json.keys())
input_arr = []

for via_key in via_json_keys:
    via_val = via_json[via_key]
    filename = via_val['filename']
    regions = via_val['regions']

    if not (args.imagesDir/filename).is_file():
        print(f"{filename} was skiped. (File not found)")
        continue

    ann_id_arr = []

    for region in regions:
        className = ""
        if "name" in region['region_attributes']:
            className = region['region_attributes']['name']
        elif "class" in region['region_attributes']:
            className = region['region_attributes']['class']
        classIds = [x['id'] for x in classArr if x['name'] == className]
        classId = classIds[0] if len(classIds) else ''

        if classId == '':
            print(f"{filename} was skipped. (Class not found in hardcoded dir)")
            continue

        shape_attr = region['shape_attributes']

        if shape_attr['name'] == "polygon" and (
                len(shape_attr['all_points_x']) < 3
                or len(shape_attr['all_points_y']) < 3):
            continue

        ann_id_arr.append(anoId)
        anoId += 1

    if len(ann_id_arr) > 0:
        for rotate_val in ROTATE_ARR:
            input_arr.append([via_val, imgId, ann_id_arr, rotate_val])
            imgId += 1

## ファイルの生成実行（並列処理）
workers = multiprocessing.cpu_count()
with multiprocessing.Pool(processes=workers) as p:
    r = p.map(generate_coco_files, input_arr)

## メタデータ（JSON）を書き出す
imageArr, annotationArr, annotationArr_pn = zip(*r)
annotationArr = flatten(annotationArr)
annotationArr_pn = flatten(annotationArr_pn)

instances_meta = {
    "info": infoArr,
    "images": imageArr,
    "licenses": licensesArr,
    "annotations": annotationArr,
    "categories": classArr
}
instances_json_path = annotations_dir/f'instances_{args.dataType}2017.json'

with instances_json_path.open('w') as f:
    json.dump(
        instances_meta, f, ensure_ascii=False, indent=4, sort_keys=True,
        separators=(',', ': '))

panoptic_meta = {
    "info": infoArr_pn,
    "images": imageArr,
    "licenses": licensesArr,
    "annotations": annotationArr_pn,
    "categories": classArr
}
panoptic_json_path = annotations_dir/f'panoptic_{args.dataType}2017.json'

with panoptic_json_path.open('w') as f:
    json.dump(
        panoptic_meta, f, ensure_ascii=False, indent=4, sort_keys=True,
        separators=(',', ': '))

captions_meta = copy.deepcopy(instances_meta)
captions_meta["annotations"] = [
        {
            "id": anno["id"],
            "image_id": anno["image_id"],
            "caption": ""
        } for anno in annotationArr
    ]
captions_json_path = annotations_dir/f'captions_{args.dataType}2017.json'

with captions_json_path.open('w') as f:
    json.dump(
        captions_meta, f, ensure_ascii=False, indent=4, sort_keys=True,
        separators=(',', ': '))
