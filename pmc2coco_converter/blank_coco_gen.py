import os 
import json 
import re 
from tqdm import tqdm
from PIL import Image

def blank_coco_gen(img_path, category_list=None):
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
    if not category_list:
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
            'bar'
        ]
    
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


if __name__ == "__main__":

    # ============ Global original data path var
    workspace = '/Users/pengyu/Desktop/Workspace/'
    root_path = workspace + "data/chart_data/pmc_coco/train/"
    annotation_path = root_path + "readable_annotations/"
    img_path = root_path + "images/"


    blank_coco = blank_coco_gen(img_path=img_path)
    open(root_path+"blank_coco.json",'w').write(json.dumps(blank_coco))