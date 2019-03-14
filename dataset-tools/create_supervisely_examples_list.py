r"""
Create a trainval.txt file listing all examples for the Supervisely persons
dataset (which consists of 13 separate datasets labelled ds1-ds13).

Note: The Supervisely dataset directory structure should be unchanged from
download.

Author: Adam Tupper
Since: 14/03/18

Example usage:
    python create_supervisely_examples_list.py \
        /media/adam/HDD\ Storage/Datasets/supervisely-persons
"""

import sys
import os

DATASET_PATH = sys.argv[1]
OUTPUT_PATH = os.path.join(DATASET_PATH, 'trainval.txt')


def main():
    if len(sys.argv) < 2:
        print('Path to dataset root directory not provided.')
        sys.exit()

    examples = []
    for dataset_dir in [f.name for f in os.scandir(DATASET_PATH) if f.is_dir()]:
        annotations_path = os.path.join(DATASET_PATH, dataset_dir, 'ann')
        examples += [os.path.join(dataset_dir, 'ann', f.name) for f in os.scandir(annotations_path)]
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write("\n".join(examples))

    print(f'{len(examples)} examples found.')
    print(f'{OUTPUT_PATH} created!')

main()