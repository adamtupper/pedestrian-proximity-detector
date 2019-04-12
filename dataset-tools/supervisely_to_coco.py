r"""
Generate a COCO style version of the Supervisely dataset.

Before running this script, trainval.txt should be generated.

Note: The Supervisely dataset directory structure should be unchanged from
download.

Author: Adam Tupper
Since: 12/04/18

Example usage:
    python supervisely_to_coco.py \
        /media/atu31/Seagate Backup Plus Drive/CV/ \
        supervisely-persons \
        trainval.txt
"""

import sys
import datetime
import json
import os
import re
import random

import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

ROOT_DIR = 'train'
IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2018")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": "Supervisely Persons Dataset",
    "url": "https://supervise.ly/",
    "version": "1.0",
    "year": 2019,
    "contributor": "Adam",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'person',
        'supercategory': 'object',
    }
]


def main():

    if len(sys.argv) < 4:
        print('Paths to root dir, trainval.txt and output dir not provided.')
        sys.exit()

    root_dir = sys.argv[1]
    dataset_dir = sys.argv[2] # Relative to root
    trainval = sys.argv[3] # Relative to root

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # Read in list of examples
    with open(os.path.join(root_dir, trainval)) as f:
        examples = [x.strip for x in f.readlines()]

    # Split examples into train and test sets
    random.seed(1)
    random.shuffle(examples)
    num_train_examples = floor(0.7 * len(examples))
    num_test_examples = len(examples) - num_train_examples
    train_examples = examples[:num_train_examples]
    test_examples = examples[num_train_examples:]
    print(f'{num_train_examples} train images. {num_test_examples} test images.')

    # Create output directories
    output_dir = dataset_dir + '-coco'
    os.mkdir(os.path.join(root_dir), output_dir)
    os.mkdir(os.path.join(root_dir), output_dir, 'annotations')
    os.mkdir(os.path.join(root_dir), output_dir, 'image-train')
    os.mkdir(os.path.join(root_dir), output_dir, 'image-test')

    # Create train examples
    # TODO: Add logs to report progress
    for example in train_examples:
        mask_found = False
        dataset, filename = example.split('/')

        # Open annotation file
        with open(os.path.join(root_dir, dataset_dir, dataset, 'ann', filename + '.json')) as f:
            annotations = json.load(f)

        image_height = annotations['size']['height']
        image_width = annotations['size']['width']

        # For each instance
        for instance in annotations['objects']:
            # If bmp mask present
            if instance['classTitle'] == 'person_bmp':

                # Create whole image mask
                mask = base64_2_mask(obj['bitmap']['data'])
                mask_origin = instance['bitmap']['origin']
                mask_height, mask_width = mask.shape

                left_pad = np.zeros((mask_height, mask_origin[0]))
                right_pad = np.zeros((mask_height, image_width - mask_width - mask_origin[0]))
                top_pad = np.zeros((mask_origin[1], image_width))
                bottom_pad = np.zeros((image_height - mask_height - mask_origin[1], image_width))

                mask = np.hstack((left_pad, mask))
                mask = np.hstack((mask, right_pad))
                mask = np.vstack((top_pad, mask))
                mask = np.vstack((mask, bottom_pad))

                # Create annotation info
                category_info = {'id': 1, 'is_crowd': 0}

                annotation_info = pycococreatortools.create_annotation_info(
                    # TODO: Specify and update IDs
                    segmentation_id,
                    image_id,
                    category_info,
                    mask,
                    (image_width, image_height),
                    tolerance=0
                )

                if annotation_info is not None:
                    coco_output['annotations'].append(annotation_info)
                    mask_found = True

        # If mask found
        if mask_found:
            # Open image
            try:
                image = Image.open(os.path.join(root_dir, dataset_dir, dataset, 'img', filename + '.png'))
            except:
                image = Image.open(os.path.join(root_dir, dataset_dir, dataset, 'img', filename + '.jpg'))

            # Save image as jpeg <dataset>_<filename> to image-train directory
            image.save(os.path.join(root_dir), output_dir, 'image-train', dataset + '_' + filename + '.jpg'), format="JPEG")

            # Create image info
            image_info = pycococreatortools.create_image_info(
                image_id,
                dataset + '_' + filename + '.jpg',
                image.size
            )

            coco_output['images'].append(image_info)

    # Save training set and clear memory

    # Create test examples

    # Save testing set and clear memory

if __name__ == "__main__":
    main()
