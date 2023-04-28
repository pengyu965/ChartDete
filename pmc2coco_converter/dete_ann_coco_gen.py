from operator import gt
import os 
import json 
import re
from tkinter import image_names
from typing import Dict, List
import numpy as np 
from PIL import Image
import multiprocessing
from tqdm import tqdm 
import cv2
import random
import math
import shutil
import sys
from blank_coco_gen import blank_coco_gen
from copy import deepcopy

category_list = [
    'x_title', 
    'y_title', 
    'plot_area', 
    'other', 
    'xlabel',
    'ylabel', 
    'chart_title', 
    'x_tick',
    'y_tick',
    'legend_patch',
    'legend_label', 
    'legend_title',
    'legend_area',
    'mark_label', 
    'value_label', 
    'y_axis_area', 
    'x_axis_area', 
    'tick_grouping', 
]

class BLANK_COCO:
    def __init__(self, img_path, category_list)->None:
        self.coco = self.blank_coco_gen(img_path,category_list)
        self.images = self.coco['images']
        self.categories = self.coco['categories']

        self.filename2img = {}
        for image in self.images:
            self.filename2img[image["file_name"]] = image

        self.label2catid={}
        for cat in self.categories:
            self.label2catid[cat['name']] = cat['id']

    def blank_coco_gen(self, img_path, category_list):
        img_file_list = os.listdir(img_path)
        images = []
        for i, img_file in enumerate(img_file_list):
            w,h = Image.open(os.path.join(img_path, img_file)).size
            images.append(
                {
                    "id": i,
                    "width": w,
                    "height": h,
                    "file_name":img_file
                }
            )
        
        catgories_ann = [
            {
                "id": i+1,
                "name": label,
                "supercategory":None
            }
            for i, label in enumerate(category_list)
        ]
        blank_coco = {'images': images, 'categories':catgories_ann}
        return blank_coco


def anndic2coco(anndic: Dict, chart_ID: str, COCO_template) -> List:
    images = COCO_template.filename2img[chart_ID+".jpg"]
    annotations = []
    for cat in anndic.keys():
        assert cat in COCO_template.label2catid.keys(), \
            "category_list didn't include all category contained in ann dict"
        for ann in anndic[cat]:
            x0,y0,x1,y1 = ann['bbox']
            w,h = x1-x0, y1-y0 
            area = w*h 

            if area == 0: #Used for filtering the bbox without highth and width
                continue

            annotations.append(
                {
                    'id': 0,
                    'image_id': images['id'],
                    'category_id': COCO_template.label2catid[cat],
                    'area': area,
                    'bbox': [x0,y0,w,h],
                    'iscrowd':0,
                    'text': ann['text'] if 'text' in ann.keys() else None
                }
            )
    return annotations

def visualize_label(label_bbox_dic, img):
    for label in label_bbox_dic.keys():
        color = (random.random()*255,random.random()*255,random.random()*255)
        for bbox_item in label_bbox_dic[label]:
            if bbox_item != () and bbox_item != [] and bbox_item != {}:
                # print(type(bbox))
                # print(label, bbox)
                bbox = bbox_item["bbox"]
                x0 = bbox[0]
                y0 = bbox[1]
                x1 = bbox[2]
                y1 = bbox[3]
                cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),color,2)
                cv2.putText(img, label, 
                    (int(x0),int(y0)), cv2.FONT_HERSHEY_PLAIN, 
                    1.2, color, 1, cv2.LINE_AA)
    return img

def task2345_process(chart_ID, j_file, img):
    def cover_bbox_gen(bbox_item_list, img=None, bbox_name=None):
        if bbox_item_list == [] or {}:
            return ()
        bbox_list = []
        # print(bbox_item_list)
        for bbox_item in bbox_item_list:
            # print(bbox_item["bbox"])
            bbox_list.append(bbox_item["bbox"])
        x0 = sorted(bbox_list, key = lambda i:i[0])[0][0]
        y0 = sorted(bbox_list, key = lambda i:i[1])[0][1]
        x1 = sorted(bbox_list, key = lambda i:i[2])[-1][2]
        y1 = sorted(bbox_list, key = lambda i:i[3])[-1][3]
        # color = (random.random()*255,random.random()*255,random.random()*255)
        # cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),color,2)
        # cv2.putText(img, bbox_name, (int(x0),int(y0)), \
        #               cv2.FONT_HERSHEY_PLAIN, 1.2, color, 1, cv2.LINE_AA)
        return (x0,y0,x1,y1)
    def polygon2bbox(polygon_dic):
        x_coords = []
        y_coords = []
        for item in polygon_dic.keys():
            if "x" in item:
                x_coords.append(polygon_dic[item])
            elif "y" in item:
                y_coords.append(polygon_dic[item])

        x0 = min(x_coords)
        y0 = min(y_coords)

        x1 = max(x_coords)
        y1 = max(y_coords)
        return (x0, y0, x1, y1)

    used_role = {'legend_title', 'chart_title', 'mark_label', 'tick_label', \
        'legend_label', 'other', 'axis_title', 'value_label', 'tick_grouping'}

    
    img = np.array(img, dtype=np.uint8)
    w,h,_ = img.shape
    

    id_text_bb_dic = {}
    text_role_dic_id = {}
    id_x_tick_dic = {}
    id_y_tick_dic = {}
    id_x2_tick_dic = {}
    id_y2_tick_dic = {}

    for role in used_role:
        text_role_dic_id[role] = []

    text_block_list = j_file["task3"]["input"]["task2_output"]["text_blocks"]
    for item in text_block_list:
        item_id = item["id"]
        polygon_dic = item["polygon"]
        # Convert the polygon to the bbox
        (bbo_x0, bbox_y0, bbo_x1, bbo_y1) = polygon2bbox(polygon_dic)
        # id_text_bb_dic[item_id] = [x0, y0, x1, y1]

        poly_x0,poly_x1,poly_x2,poly_x3,\
            poly_y0,poly_y1,poly_y2,poly_y3 = polygon_dic.values()
        id_text_bb_dic[item_id] = {
            "bbox":[bbo_x0, bbox_y0, bbo_x1, bbo_y1], 
            "polygon":[poly_x0,poly_x1,poly_x2,poly_x3,
                        poly_y0,poly_y1,poly_y2,poly_y3],
            "text":item['text']
            }

    text_role_list = j_file["task3"]["output"]["text_roles"]
    for item in text_role_list:
        role = item["role"]
        item_id = item["id"]
        # if role in used_role:
        if role not in text_role_dic_id.keys():
            text_role_dic_id[role] = []
        text_role_dic_id[role].append(item_id) 

    tick_dic = j_file["task4"]["output"]["axes"]
    for item in tick_dic["x-axis"]:
        item_id = item["id"]
        x_c, y_c = item["tick_pt"].values()

        x0 = x_c - 5
        y0 = y_c - 5
        x1 = x_c + 5
        y1 = y_c + 5

        id_x_tick_dic[item_id] = {"pts": [x_c,y_c], "bbox":[x0,y0,x1,y1]}


    for item in tick_dic["y-axis"]:
        item_id = item["id"]
        x_c, y_c = item["tick_pt"].values()

        x0 = x_c - 5
        y0 = y_c - 5
        x1 = x_c + 5
        y1 = y_c + 5

        id_y_tick_dic[item_id] = {"pts": [x_c,y_c], "bbox":[x0,y0,x1,y1]}

    # =====
    # In future, Consider the x2 axis and y2 axis

    plot_area_bb = [
        j_file["task4"]["output"]["_plot_bb"]["x0"],
        j_file["task4"]["output"]["_plot_bb"]["y0"],
        j_file["task4"]["output"]["_plot_bb"]["x0"] 
            + j_file["task4"]["output"]["_plot_bb"]["width"],
        j_file["task4"]["output"]["_plot_bb"]["y0"] 
            + j_file["task4"]["output"]["_plot_bb"]["height"]
    ]


    x_axis_area_bb_list = []
    y_axis_area_bb_list = []
    legend_area_bb_list = []

    x_title_bb_list = []
    y_title_bb_list = []
    x_label_bb_list = []
    y_label_bb_list = []

    legend_patch_list = []



    # print(text_role_dic_id)
    # print(id_text_bb_dic)

    if "tick_label" in text_role_dic_id.keys():
        for iid in id_x_tick_dic.keys():
            # Append the tick mark to axis area
            x_axis_area_bb_list.append(id_x_tick_dic[iid]) 
            # Tick Label separate
            if iid in text_role_dic_id["tick_label"]:
                # print(len(id_text_bb_dic[iid]),"fff")
                # Append the tick label to axis area
                x_axis_area_bb_list.append(id_text_bb_dic[iid]) 
                # Save the seperated tick label
                x_label_bb_list.append(id_text_bb_dic[iid]) 

    if "tick_label" in text_role_dic_id.keys():
        for iid in id_y_tick_dic:
            # Append the tick mark to axis area
            y_axis_area_bb_list.append(id_y_tick_dic[iid]) 
            # Tick Label separate
            if iid in text_role_dic_id["tick_label"]:
                # Append the tick label to axis area
                y_axis_area_bb_list.append(id_text_bb_dic[iid])
                # Save the seperated tick label
                y_label_bb_list.append(id_text_bb_dic[iid]) 


    if "axis_title" in text_role_dic_id.keys():
        # Distinguish Axis title by x_axis_title and y_axis_title
        for axis_title_id in text_role_dic_id["axis_title"]:
            text_bb_item = id_text_bb_dic[axis_title_id]
            bbox = text_bb_item["bbox"]
            title_bb_center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]

            # Right now, the x axis area only include tick label and tick mark
            x_labels_tick_area = cover_bbox_gen(x_axis_area_bb_list)
            # Right now, the y axis area only include tick label and tick mark
            y_labels_tick_area = cover_bbox_gen(y_axis_area_bb_list) 

            if x_labels_tick_area != ():
                x_labels_center = \
                    [(x_labels_tick_area[0]+x_labels_tick_area[2])/2, 
                        (x_labels_tick_area[1]+x_labels_tick_area[3])/2]
                title_x_labels_dis = \
                    math.sqrt((title_bb_center[0]-x_labels_center[0])**2 + 
                                (title_bb_center[1]-x_labels_center[1])**2)
            else:
                title_x_labels_dis = float("inf")

            if y_labels_tick_area != ():
                y_labels_center = \
                    [(y_labels_tick_area[0]+y_labels_tick_area[2])/2, 
                        (y_labels_tick_area[1]+y_labels_tick_area[3])/2]
                title_y_labels_dis = \
                    math.sqrt((title_bb_center[0]-y_labels_center[0])**2 
                                + (title_bb_center[1]-y_labels_center[1])**2)
            else:
                title_y_labels_dis = float("inf")

            # xy_ratio = (bbox[2]-bbox[0])/(bbox[3]-bbox[1])
            if title_x_labels_dis < title_y_labels_dis:
                x_axis_area_bb_list.append(text_bb_item)
                x_title_bb_list.append(text_bb_item)
            else:
                y_axis_area_bb_list.append(text_bb_item)
                y_title_bb_list.append(text_bb_item)

    # Processing legend_patch
    for item in j_file["task5"]["output"]["legend_pairs"]:
        x0 = item["bb"]["x0"]
        y0 = item["bb"]["y0"]
        x1 = x0 + item["bb"]["width"]
        y1 = y0 + item["bb"]["height"]

        legend_patch_list.append({"bbox":[x0,y0,x1,y1]})

    
    for legend_id in text_role_dic_id["legend_title"] + text_role_dic_id["legend_label"]:
        legend_area_bb_list.append(id_text_bb_dic[legend_id])
    for bbox_item in legend_patch_list:
        legend_area_bb_list.append(bbox_item)


    x_axis_area_bb = cover_bbox_gen(x_axis_area_bb_list, img, "x_axis_area")
    y_axis_area_bb = cover_bbox_gen(y_axis_area_bb_list, img, "y_axis_area")
    legend_area_bb = cover_bbox_gen(legend_area_bb_list, img, "legend_area")

    # print(len(text_role_dic_id["tick_label"]))
    # print(len(x_label_bb_list))
    # print(len(y_label_bb_list))
    
    area_bb_gt_dic ={
        "x_axis_area":[{"bbox": x_axis_area_bb}] if x_axis_area_bb != () else [],
        "y_axis_area":[{"bbox": y_axis_area_bb}] if y_axis_area_bb != () else [],
        "plot_area":[{"bbox": plot_area_bb}] if plot_area_bb != () else [],
        "legend_area": [{"bbox": legend_area_bb}] if legend_area_bb != () else [],
        "x_title":x_title_bb_list,
        "y_title":y_title_bb_list,
        "xlabel":x_label_bb_list,
        "ylabel":y_label_bb_list,
        "legend_patch": legend_patch_list,
        "x_tick": list(id_x_tick_dic.values()),
        "y_tick": list(id_y_tick_dic.values())
    }

    label_gt_dic = {}
    for label_role in text_role_dic_id.keys():
        if label_role not in label_gt_dic.keys():
            label_gt_dic[label_role] = []
        for text_bb_id in text_role_dic_id[label_role]:
            label_gt_dic[label_role].append(id_text_bb_dic[text_bb_id])


    if "tick_label" in label_gt_dic.keys(): label_gt_dic.pop("tick_label")
    if "axis_title" in label_gt_dic.keys(): label_gt_dic.pop("axis_title")

    area_bb_gt_dic.update(label_gt_dic)

    # Correct the bbox that has negative number
    # correction_dic(area_bb_gt_dic)

    # cv2.imshow("structrural result",img)
    # cv2.waitKey()
   
    return area_bb_gt_dic

def task6_bar(j_file):
    bar_bboxes = []
    for bb in j_file['task6']['output']['visual elements']['bars']:
        p1 = (int(bb['x0']), int(bb['y0']))
        p2 = (int(bb['x0'] + bb['width']), int(bb['y0'] + bb['height']))
        
        bbox = [p1[0], p1[1], p2[0], p2[1]]

        # Check whether the second point coords 
        # isn't larger than the first point
        # if (p1[0] >= p2[0]) or (p1[1] >= p2[1]):
        #     print(p1)
        #     print(p2)
        #     print(chart_folder)
        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
            continue

        bar_bboxes.append({"bbox": bbox})
    return bar_bboxes

def task6_boxplot(j_file):
    boxplot_bboxes = []
    for item in j_file['task6']['output']['visual elements']['boxplots']:
        min_bb = item['min']['_bb']
        max_bb = item['max']['_bb']

        if int(min_bb["height"]) == 0:
            x0 = min_bb["x0"]
            x1 = x0 + min_bb["width"]
            y0 = min(min_bb["y0"], max_bb["y0"])
            y1 = max(min_bb["y0"], max_bb["y0"])
        elif int(min_bb["width"]) == 0:
            y0 = min_bb["x0"]
            y1 = x0 + min_bb["height"]
            x0 = min(min_bb["x0"], max_bb["x0"])
            x1 = max(min_bb["x0"], max_bb["x0"])
        else:
            print(item)

        if x0 == x1 or y0 == y1:
            continue

        boxplot_bboxes.append({"bbox": [x0,y0,x1,y1]})
        # print(boxplot_bboxes)
    return boxplot_bboxes

def coco_filter_imgname(coco, imgname_set):
    imgname_set = set(imgname_set)
    images = []
    anns = []
    imgid2imgname = {}

    # Filter the images and annotations in coco by imgname_set
    for image in coco["images"]:
        imgid2imgname[image["id"]] = image["file_name"]
        if image["file_name"] in imgname_set:
            images.append(image)

    for ann in coco["annotations"]:
        if imgid2imgname[ann["image_id"]] in imgname_set:
            anns.append(ann)
    
    imgid_old2new = {}
    # Change image["id"] and ann["id"] to sequential from 0-N
    for i, image in enumerate(images):
        imgid_old2new[image["id"]] = i 
        images[i]["id"] = i 

    for i, ann in enumerate(anns):
        ann["id"] = i
        ann["image_id"] = imgid_old2new[ann["image_id"]]
    
    new_coco = {
        "images":images,
        "annotations":anns,
        "categories":coco["categories"]}
    
    return new_coco


# In future, add all the paths like workspace, task_root etc as 
# the paratmeter of the function for better controlling 
def handle_train_val_dataset(include_plots=[]):
    workspace = '/home/csgrad/pyan4/Workspace/'
    # origin_pmc = workspace + "data/chart_data/original_pmc/"
    task_root = \
            workspace + "data/chart_data/pmc_2022/pmc_coco/element_detection/"
    image_path = task_root + "bar_images/"

    pmc_ann = task_root + "pmc_ori_ann/bar_charts/"
    
    for cate in include_plots:
        category_list.append(cate)

    blank_coco_template = BLANK_COCO(image_path, category_list=category_list)

    annotations = []

    for j_file in tqdm(os.listdir(pmc_ann)):
        chart_ID = os.path.splitext(j_file)[0]
        img = Image.open(image_path + chart_ID + ".jpg")
        img = img.convert("RGB")
        pmc_ann = json.load(open(pmc_ann + j_file,'r'))

        element_object_dic = task2345_process(chart_ID, pmc_ann, img)
        # So far, only handled bar, if more plots element need to be included, 
        # then add code accordingly as following
        if 'bar' in include_plots:
            bar_bboxes = task6_bar(pmc_ann)
            element_object_dic.update({"bar":bar_bboxes})

        annotations += anndic2coco(element_object_dic, 
                                    chart_ID, blank_coco_template)

    
    blank_coco_template.coco["annotations"] = annotations

    barchart_id_l = os.listdir(pmc_ann)

    train_barchart_id_l = random.sample(barchart_id_l, 
                                        round(0.9*len(barchart_id_l)))
    val_barchart_id_l = [x for x in barchart_id_l 
                            if x not in train_barchart_id_l]
    
    train_imgnames = [
        x[:-5] + ".jpg" 
        for x in train_barchart_id_l]
    val_imgnames = [
        x[:-5] + ".jpg" 
        for x in val_barchart_id_l]

    train_coco = coco_filter_imgname(
        deepcopy(blank_coco_template.coco), train_imgnames)
    val_coco = coco_filter_imgname(
        deepcopy(blank_coco_template.coco), val_imgnames)

    with open(task_root+"bar_train.json",'w')as f:
        f.write(json.dumps(train_coco))
    with open(task_root+"bar_val.json",'w') as f: 
        f.write(json.dumps(val_coco))


# In future, add all the paths like workspace, task_root etc as 
# the paratmeter of the function for better controlling 
def handle_test_dataset(include_plots=[]):
    workspace = '/home/pengyu/Workspace/'
    # origin_pmc = workspace + "data/chart_data/original_pmc/"
    pmc_root = workspace + "data/chart_data/pmc_2022/"

    task_root = workspace + "data/chart_data/pmc_2022/pmc_coco/element_detection/"

    test_img_dest = task_root + "split3_test/"
    os.makedirs(test_img_dest, exist_ok=True)

    test_img_ori = pmc_root + "test_with_GT/split_3/images/"
    test_gt_ori = pmc_root + "test_with_GT/split_3/annotations/"

    for cate in include_plots:
        category_list.append(cate)

    blank_coco_template = BLANK_COCO(test_img_dest, category_list=category_list)

    annotations = []

    for image in tqdm(os.listdir(test_img_dest)):
        chart_ID = os.path.splitext(image)[0]
        img = Image.open(test_img_dest + chart_ID + ".jpg")
        img = img.convert("RGB")
        pmc_ann = json.load(open(test_gt_ori + chart_ID + ".json",'r'))

        element_object_dic = task2345_process(chart_ID, pmc_ann, img)
        if 'bar' in include_plots:
            bar_bboxes = task6_bar(pmc_ann)
            element_object_dic.update({"box_plot":bar_bboxes})

        annotations += anndic2coco(element_object_dic, 
                                    chart_ID, blank_coco_template)

        # Following exit is only for test debug use
        # sys.exit()


    for i in range(len(annotations)):
        annotations[i]['id'] = i 

    blank_coco_template.coco["annotations"] = annotations
    open(task_root+"split3_test.json",'w').write(
        json.dumps(blank_coco_template.coco))


if __name__ == "__main__":
    handle_train_val_dataset()
    handle_test_dataset()
