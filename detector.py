r"""
Pedestrian proximity detector using Mask R-CNN and an Intel RealSense D435 RGB-D camera.

If the path to an RGB-D data bag is not given, the program will attempt to load a live stream from a
connected camera. When streaming from a bag, the program will loop through the footage.

Author: Adam Tupper
Since: 08/04/19

Example usage:
    python detector.py \
        --model-config [config.yaml] \
        --confidence-threshold 0.7 \
        --video-file [input.bag] \
        --output-file [video.avi]
"""

import sys
import os
import argparse

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from PIL import Image
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark import layers as L
from predictor import PedestrianPredictor
import matplotlib.pyplot as plt

# Specify program arguments
parser = argparse.ArgumentParser(description="Pedestrian Proximiy Detector")
parser.add_argument(
    "--model-config",
    default="models/mask_rcnn_resnet_50_supervisely/trained_supervisely_config.yaml",
    metavar="FILE",
    help="Path to the YAML config file for the trained Mask R-CNN model",
)
parser.add_argument(
    "--confidence-threshold",
    type=float,
    default=0.7,
    help="Minimum score for the prediction to be shown",
)
parser.add_argument(
    "--video-file",
    metavar="FILE",
    help="A video file to process instead of a live camera feed",
)
parser.add_argument(
    "--output-file",
    metavar="FILE",
    help="A path to save the annotated video file to",
)


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


def main():
    # Parse command-line arguments
    args = parser.parse_args()

    # Load model config from file
    cfg.merge_from_file(args.model_config)
    cfg.freeze()

    # Prepare model predictor
    predictor = PedestrianPredictor(
        cfg,
        confidence_threshold=args.confidence_threshold
    )

    # Setup video writer (if required)
    if args.output_file is not None:
        print('Saving annotated stream to', args.output_file)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(args.output_file, fourcc, 30, (1280, 480))

    # Configure RealSense camera
    config = rs.config()

    if args.video_file is None:
        print('Streaming data from camera...')
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    else:
        print('Streaming data from file...')
        config.enable_device_from_file(args.video_file)

    pipeline = rs.pipeline() # Create a pipline
    profile = pipeline.start(config) # Start streaming
    align = rs.align(rs.stream.color) # Align infrared and depth images to colour image

    # Get depth sensor and scale information
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

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
        segmented_image = predictor.run_on_opencv_image(color_image, depth_image * depth_scale)

        output_image = np.hstack((segmented_image, depth_colormap))
        cv.imshow('Pedestrian Proximity Detector', output_image)

        if args.output_file is not None:
            out.write(output_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            # Quit if 'q' key is pressed
            break

    pipeline.stop() # Stop recording
    cv.destroyAllWindows() # Close all windows
    if args.output_file is not None:
        out.release() # Release output file (if required)


if __name__ == '__main__':
    main()
