r"""
Pedestrian proximity detector using Mask R-CNN and an Intel RealSense D435 RGB-D camera.

If the path to an RGB-D data bag is not given, the program will attempt to load a live stream from a
connected camera. When streaming from a bag, the program will loop through the footage.

Author: Adam Tupper
Since: 08/04/19

Example usage:
    python detector.py [input.bag]
"""

import sys
import os

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import tensorflow as tf
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

# Mask R-CNN model configuration (must use absolute paths)
MODEL_NAME = 'mask_rcnn_inception_v2_supervisely_2018_03_31'
PATH_TO_CKPT = os.path.join(os.getcwd(), 'models', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(os.getcwd(), 'data', 'persons_label_map.pbtxt')
NUM_CLASSES = 1


def load_image_into_numpy_array(image):
    """
    Load a PIL image into a numpy array.

    Args:
        image: A PIL image.

    Returns:
        array: A 2D numpy array of the image.
    """
    (im_width, im_height) = image.size
    array = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    return array


def run_inference_for_single_image(image, graph):
    """
    Perform instance segmentation on an image using our model.

    Args:
        image: A 2D numpy array of the image.
        graph: The instance segmentation model graph.

    Returns:
        output_dict: A dictionary containing model outputs, including the predicted instance masks.
    """
    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config=config)
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def main():
    # Load trained Mask R-CNN model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Load model label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
                     label_map,
                     max_num_classes=NUM_CLASSES,
                     use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Configure RealSense camera
    config = rs.config()

    if len(sys.argv) > 1:
        print('Streaming data from file...')
        path_to_bag = sys.argv[1]
        config.enable_device_from_file(path_to_bag)
    else:
        print('Streaming data from camera...')
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline() # Create a pipline
    profile = pipeline.start(config) # Start streaming
    align = rs.align(rs.stream.color) # Align infrared and depth images to colour image

    while True:
        frames = pipeline.wait_for_frames() # Read images from camera
        aligned_frames = align.process(frames)

        # Get aligned frames as Numpy arrays
        depth_image = np.asanyarray(aligned_frames.get_depth_frame().get_data())
        color_image = np.asanyarray(aligned_frames.get_color_frame().get_data())

        # Colourise depth map for viewing
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03),
                                          cv.COLORMAP_JET)

        # Perform instance segmentation on RGB image
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        color_image_expanded = np.expand_dims(color_image, axis=0)
        output_dict = run_inference_for_single_image(color_image_expanded, detection_graph)

        masks = output_dict.get('detection_masks')
        if len(masks) > 0:
            h, w = masks[0].shape
            mask = masks[0].reshape(h, w)
            cv.imshow('Segmentation', mask)

        cv.imshow('Colour', color_image)
        cv.imshow('Depth', depth_colormap)

        if cv.waitKey(1) & 0xFF == ord('q'):
            # Quit if 'q' key is pressed
            break

    pipeline.stop() # Stop recording
    cv.destroyAllWindows() # Close all windows


if __name__ == '__main__':
    main()
