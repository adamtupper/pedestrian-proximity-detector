# Pedestrian Proximity Detector

## Abstract

This paper presents a novel method for pedestrian detection and distance estimation using RGB-D data. We use Mask R-CNN for instance-level pedestrian segmentation, and the Semiglobal Matching algorithm for computing depth information from a pair of infrared images captured by an Intel RealSense D435 stereo vision depth camera. The resulting depth map is post-processed using both spatial and temporal edge-preserving filters and spatial hole-filling to mitigate erroneous or missing depth values. The distance to each pedestrian is estimated using the median depth value of the pixels in the depth map covered by the predicted mask. Unlike previous work, our method is evaluated on, and performs well across, a wide spectrum of outdoor lighting conditions. Our proposed technique is able to detect and estimate the distance of pedestrians within 5m with an average accuracy of 87.7\%.

## Intel RealSense D435 Setup

Instructions for installing the Intel RealSense SDK can be found [here](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md).

## Running the detector

From the `pedestrian-proximity-detector` directory, execute:

```
python detector.py \
    --model-config [config.yaml] \
    --confidence-threshold 0.7 \
    --video-file [input.bag] \
    --output-file [video.avi]
```

All parameters apart from the model config are optional. The default value for the confidence threshold is 0.7. If no video file is provided, the detector will attempt to stream from a connect D435 camera.

## Converting the Supervisely dataset into COCO format

From the `pedestrian-proximity-detector/dataset-tools` directory execute:

```
python create_supervisely_examples_list.py [path to dataset directory]
```

This will create a `trainval.txt` file containg the list of examples (images) in the Supervisely Persons dataset. Then, from the `pedestrian-proximity-detector/dataset-tools` directory, execute:

```
python supervisely_to_coco.py \
    [root directory] \
    [path from root to dataset directory] \
    trainval.txt
```

The first arguement is the path to the directory which holds the Supervisely Persons dataset, the second arguement is the directory containing the Supervisely Persons dataset (in the format/structure downloaded from supervisely) and the is the trainval.txt examples file generated inn the previous step.

## Training the Pedestrian Segmentation (Mask R-CNN) Model

### 1. Fetch the  `maskrcnn-benchmark` repository

If not already downloaded, clone the [maskrcnn-benchmark](https://github.com/adamtupper/maskrcnn-benchmark) repository in the parent directory of this directory and checkout the `supervisely` branch. You should have the following directory structure:

```
maskrcnn-benchmark/
pedestrian-proximity-detector/
```

### 3. Install the `maskrcnn_benchmark` Python package

From the `maskrcnn-benchmark directory`, execute:

```
python setup.py install
```

Note: The package must be reinstalled if anything in the `maskrcnn_benchmark` directory is changed.

### 2. Remove the final layers from the pre-trained COCO model

From the `pedestrian-proximity-detector/model-tools` directory, excute the following:

```
python trim_cnn_layers.py
```

This will create a PyTorch model file with no final layers in the `pedestrian-proximity-detector/models/mask_rcnn_resnet_50_supervisely/` directory.

### 3. Training the model

From inside the `maskrcnn-benchmark directory` execute:

```
python tools/train_net.py --config-file ../pedestrian-proximity-detector/models/mask_rcnn_resnet_50_supervisely/supervisely_config.yaml
```

The trained model file will be placed in the `pedestrian-proximity-detector/models/mask_rcnn_resnet_50_supervisely/` directory.

## Evaluate the trained model on COCO dataset

From the `maskrcnn-benchmark` directory, run:

```
python tools/test_net.py --config-file ../pedestrian-proximity-detector/models/mask_rcnn_resnet_50_supervisely trained_supervisely_coco_eval_config.yaml
```
