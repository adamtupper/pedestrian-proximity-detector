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

import pyrealsense2 as rs
import numpy as np
import cv2 as cv


def main():
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

        cv.imshow('Colour ', color_image)
        cv.imshow('Depth', depth_colormap)

        if cv.waitKey(1) & 0xFF == ord('q'):
            # Quit if 'q' key is pressed
            break

    pipeline.stop() # Stop recording
    cv.destroyAllWindows() # Close all windows


if __name__ == '__main__':
    main()
