"""read vwww coco annotations, output to a directory with
0 and 1. 1 means person, 0 means no person, the images are linked to the coco images"""

from pycocotools.coco import COCO
import os, sys
import shutil

annotations_file = sys.argv[1]
out_dir = sys.argv[2]
image_dir='/datasets01/COCO/022719'

foreground_class_name = 'person'
coco = COCO(annotations_file)
foreground_class_id = 1
dataset = coco.dataset

images = dataset['images']
# Create category index
foreground_category = None
background_category = {'supercategory': 'background', 'id': 0, 'name': 'background'}
for category in dataset['categories']:
    if category['name'] == foreground_class_name:
        foreground_class_id = category['id']
        foreground_category = category
foreground_category['id'] = 1
background_category['name'] = "not-{}".format(foreground_category['name'])
categories = [background_category, foreground_category]

if not 'annotations' in dataset:
    raise KeyError('Need annotations in json file to build the dataset.')
new_ann_id = 0
annotations = []
positive_img_ids = set()
foreground_imgs_ids = coco.getImgIds(catIds=foreground_class_id)
pos_dir = os.path.join(out_dir, '1')
neg_dir = os.path.join(out_dir, '0')
os.makedirs(pos_dir, exist_ok=True)
os.makedirs(neg_dir, exist_ok=True)
for img in images:
    positive = img['id'] in foreground_imgs_ids
    url = img['coco_url']
    file = url.replace('http://images.cocodataset.org', image_dir)
    assert os.path.exists(file), file
    link_path = os.path.join(pos_dir if positive else neg_dir, os.path.basename(file))
    os.symlink(file, link_path)

    
