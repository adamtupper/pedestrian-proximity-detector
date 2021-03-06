r"""
Capture RGB-D data from Intel RealSense D435 camera and save to disk.

Author: Adam Tupper (adapted from COSC428 Lab 3 code)
Since: 06/04/19

Example usage:
    python d435_capture.py output.bag
"""

import sys

import pyrealsense2 as rs
import numpy as np
import cv2 as cv


def main():
    if len(sys.argv) < 2:
        print('Output file path not provided. Exiting...')
        sys.exit()

    path_to_bag = sys.argv[1]

    # Configure RealSense camera
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_record_to_file(path_to_bag)

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
