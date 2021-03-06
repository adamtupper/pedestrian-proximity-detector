r"""
Convert the Supervisely persons dataset to TFRecord for semantic
segmentation.

More information about using the TensorFlow Object Detection API under
research at https://github.com/tensorflow/models/

Author: Adam Tupper (adapted from TensorFlow documentation)
Since: 12/03/18

Example usage:
    python create_supervisely_tf_record.py \
        --data_dir='/media/adam/HDD Storage/Datasets/supervisely-persons' \
        --output_dir='/media/adam/HDD Storage/Datasets/supervisely-persons-640x480-tf-records'
"""

import hashlib
import io
import logging
import os
import random
import re
import json
import zlib
import base64

import cv2 as cv
import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw persons dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '../data/persons_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('bounding_only', False, 'If True, generates bounding boxes '
                     'for each person.  Otherwise generates bounding boxes (as '
                     'well as segmentations for each person).  Note that '
                     'in the latter case, the resulting files are much larger.')
flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
                    'segmentation masks. Options are "png" or "numerical".')
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')
flags.DEFINE_integer('max_img_width', 640, 'The maximum width of images. Wider images will be'
                     'centrally cropped.')
flags.DEFINE_integer('max_img_height', 480, 'The maximum height of images. Taller images will be'
                     'centrally cropped.')

FLAGS = flags.FLAGS

logging.basicConfig(level=logging.INFO)


def base64_2_mask(s):
    """
    Convert from a base64 encoded string to numpy mask. Provided by Supervisely,
    see https://docs.supervise.ly/ann_format/.

    Args:
        s: A base64 encoded string of the image mask.

    Returns:
        mask: A 2D numpy array of the image mask.
    """
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv.imdecode(n, cv.IMREAD_UNCHANGED)[:, :, 3].astype(np.uint8)

    return mask


def mask_2_base64(mask):
    """
    Convert from a numpy mask to a base64 encoded string. Provided by
    Supervisely, see https://docs.supervise.ly/ann_format/.

    Args:
        mask: 2D numpy array of the image mask.

    Returns:
        s: A base64 encoded string of the image mask.
    """
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0,0,0,255,255,255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()

    s = base64.b64encode(zlib.compress(bytes)).decode('utf-8')
    return s


def crop_center(image, width, height):
    """
    Takes an image as a numpy array and crops to the centre region.

    Args:
        image: A numpy array of the image.
        width: The desired width of the cropped image.
        height: The desired height of the cropped image.

    Returns:
        image_crop: A numpy array of the cropped image.
    """
    y, x = image.shape[:2]

    if y > height:
        start_y = y // 2 - (height // 2)
        image = image[start_y:start_y + height, :]

    if x > width:
        start_x = x // 2 - (width // 2)
        image = image[:, start_x:start_x + width]

    return image


def dict_to_tf_example(data,
                       label_map_dict,
                       image_filename,
                       image_subdirectory,
                       bounding_only=True,
                       mask_type='png'):
    """
    Convert JSON derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
        data: dict holding the Supervisely annotation fields (with masks) for
            the image. For more information see
            https://docs.supervise.ly/ann_format/.
        label_map_dict: A map from string label names to integers ids.
        image_filename: String specifying the name of the image.
        image_subdirectory: String specifying subdirectory within the
            persons dataset directory holding the actual image data.
        bounding_only: If True, generates bounding boxes for each person.
            Otherwise generates bounding boxes (as well as segmentations for
            each person).
        mask_type: 'numerical' or 'png'. 'png' is recommended because it leads
        to smaller file sizes.

    Returns:
        example: The converted tf.Example.

    Raises:
        ValueError: if the image pointed to by image_filename is not a valid PNG.
    """
    x_mins = []
    y_mins = []
    x_maxs = []
    y_maxs = []
    classes = []
    classes_text = []
    masks = []
    image_format = 'png' # If image is JPEG it is converted to PNG

    # Identify image extension
    if os.path.exists(os.path.join(image_subdirectory, image_filename + '.jpg')):
        image_extension = '.jpg'
    elif os.path.exists(os.path.join(image_subdirectory, image_filename + '.png')):
        image_extension = '.png'
    else:
        raise ValueError('Image must be in PNG or JPEG format')

    # Load the image
    image_path = os.path.join(image_subdirectory, image_filename + image_extension)
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()
    encoded_image_io = io.BytesIO(encoded_image)
    image = PIL.Image.open(encoded_image_io)
    if not image.format in ['JPEG', 'PNG']:
        raise ValueError('Image must be in PNG or JPEG format')

    # Convert cropped image back to byte string
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='PNG')
    encoded_image = bytes_io.getvalue()

    key = hashlib.sha256(encoded_image).hexdigest()

    # Crop image
    image_width_og, image_height_og = image.size

    if image_width_og > FLAGS.max_img_width:
        left = (image_width_og - FLAGS.max_img_width) // 2
        right = (image_width_og + FLAGS.max_img_width) // 2
        image = image.crop((left, 0, right, image.size[1]))
    else:
        # Still needed for adjusting bounding box
        left = 0

    if image_height_og > FLAGS.max_img_height:
        top = (image_height_og - FLAGS.max_img_height) // 2
        bottom = (image_height_og + FLAGS.max_img_height) // 2
        image = image.crop((0, top, image.size[0], bottom))
    else:
        # Still needed for adjusting bounding box
        top = 0

    image_width_crop, image_height_crop = image.size

    # Extract object data
    if 'objects' in data:
        for obj in data['objects']:
            if obj['classTitle'] == 'person_bmp': # Use only objects with masks
                mask = base64_2_mask(obj['bitmap']['data'])

                # Get original mask location and dimensions
                mask_origin_og = obj['bitmap']['origin']
                mask_height_og = mask.shape[0]
                mask_width_og = mask.shape[1]

                # Update mask location and dimensions for cropped image
                mask_origin_crop = (max(mask_origin_og[0] - left, 0),
                                    max(mask_origin_og[1] - top, 0))
                mask_height_crop = min(mask_height_og, image_height_crop - mask_origin_crop[1])
                mask_width_crop = min(mask_width_og, image_width_crop - mask_origin_crop[0])

                # Obtain bounding box coords
                x_min = float(mask_origin_crop[0])
                x_max = float(mask_origin_crop[0] + mask_width_crop)
                y_min = float(mask_origin_crop[1])
                y_max = float(mask_origin_crop[1] + mask_height_crop)

                # Normalise bounding box coords
                x_mins.append(x_min / image_width_crop)
                y_mins.append(y_min / image_height_crop)
                x_maxs.append(x_max / image_width_crop)
                y_maxs.append(y_max / image_height_crop)

                class_name = 'Person'
                classes_text.append(class_name.encode('utf8'))
                classes.append(label_map_dict[class_name])

                if not bounding_only:
                    # Create whole-image mask
                    left_pad = np.zeros((mask_height_og, mask_origin_og[0]))
                    right_pad = np.zeros((mask_height_og,
                                          image_width_og - mask_width_og - mask_origin_og[0]))
                    top_pad = np.zeros((mask_origin_og[1], image_width_og))
                    bottom_pad = np.zeros((image_height_og - mask_height_og - mask_origin_og[1],
                                           image_width_og))

                    mask = np.hstack((left_pad, mask))
                    mask = np.hstack((mask, right_pad))
                    mask = np.vstack((top_pad, mask))
                    mask = np.vstack((mask, bottom_pad))
                    mask = crop_center(mask, FLAGS.max_img_width, FLAGS.max_img_height)

                    # Check mask dimensions match image
                    if mask.shape != (image_height_crop, image_width_crop):
                        logging.error('Mask and image dimensions do not match.')
                        raise ValueError('Aborting, mask and image dimensions do not match')

                    masks.append(mask)

    feature_dict = {
        'image/height': dataset_util.int64_feature(image_height_crop),
        'image/width': dataset_util.int64_feature(image_width_crop),
        'image/filename': dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(x_mins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(x_maxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(y_mins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(y_maxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }

    if not bounding_only:
        # Add segmentation masks
        if mask_type == 'numerical':
            mask_stack = np.stack(masks).astype(np.float32)
            masks_flattened = np.reshape(mask_stack, [-1])
            feature_dict['image/object/mask'] = (
                dataset_util.float_list_feature(masks_flattened.tolist())
            )
        elif mask_type == 'png':
            encoded_mask_png_list = []
            for mask in masks:
                img = PIL.Image.fromarray(mask).convert('1')
                output = io.BytesIO()
                img.save(output, format='PNG')
                encoded_mask_png_list.append(output.getvalue())
                feature_dict['image/object/mask'] = (
                    dataset_util.bytes_list_feature(encoded_mask_png_list)
                )

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     data_dir,
                     examples,
                     bounding_only=True,
                     mask_type='png'):
    """
    Creates a TFRecord file from examples.

    Args:
        output_filename: Path to where output file is saved.
        num_shards: Number of shards for output file.
        label_map_dict: The label map dictionary.
        data_dir: Root directory of the dataset.
        examples: Examples to parse and save to tf record.
        bounding_only: If True, generates bounding boxes for each person.  Otherwise
            generates bounding boxes (as well as segmentations for each person).
        mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
            smaller file sizes.
    """
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filename, num_shards
        )

        for idx, example in enumerate(examples):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples))

            # Split example name from sub-dataset
            example_dataset, example = example.split('/')
            annotations_dir = os.path.join(data_dir, example_dataset, 'ann')
            image_dir = os.path.join(data_dir, example_dataset, 'img')
            json_path = os.path.join(annotations_dir, example + '.json')

            if not os.path.exists(json_path):
                logging.warning('Could not find %s, ignoring example.', json_path)
                continue
            with tf.gfile.GFile(json_path, 'r') as fid:
                data = json.load(fid)

            try:
                tf_example = dict_to_tf_example(
                    data,
                    label_map_dict,
                    example,
                    image_dir,
                    bounding_only=bounding_only,
                    mask_type=mask_type
                )
                if tf_example:
                  shard_idx = idx % num_shards
                  output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            except ValueError:
                logging.warning('Invalid example: %s, ignoring.', json_path)


def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from supervisely-persons dataset.')
    examples_path = os.path.join(data_dir, 'trainval.txt')
    examples_list = dataset_util.read_examples_list(examples_path)

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir,
                                     'persons_bounding_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'persons_bounding_val.record')
    if not FLAGS.bounding_only:
        train_output_path = os.path.join(FLAGS.output_dir,
                                         'persons_with_masks_train.record')
        val_output_path = os.path.join(FLAGS.output_dir,
                                       'persons_with_masks_val.record')
    create_tf_record(
        train_output_path,
        FLAGS.num_shards,
        label_map_dict,
        data_dir,
        train_examples,
        bounding_only=FLAGS.bounding_only,
        mask_type=FLAGS.mask_type
    )
    create_tf_record(
        val_output_path,
        FLAGS.num_shards,
        label_map_dict,
        data_dir,
        val_examples,
        bounding_only=FLAGS.bounding_only,
        mask_type=FLAGS.mask_type
    )


if __name__ == '__main__':
    tf.app.run()
