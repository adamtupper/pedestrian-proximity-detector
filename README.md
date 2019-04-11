# pedestrian-proximity-detector

Download links for pretrained Tensorflow Object Detection API models can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

1. Generate TensorFlow records for Supervisely Persons dataset:

```
python create_supervisely_tf_record.py --data_dir='/media/adam/HDD Storage/Datasets/supervisely-persons' --output_dir='/media/adam/HDD Storage/Datasets/supervisely-persons-640x480-tf-records'
```

2. Train Mask R-CNN model:

```
python train.py --logtostderr --train_dir=/media/adam/HDD\ Storage/Datasets/supervisely-persons-640x480-tf-records ^ --pipeline_config_path=/home/adam/Development/pedestrian-proximity-detector/models/mask_rcnn_inception_v2_coco_2018_01_28/pipeline.config
```

3. Export inference graph:

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/adam/Development/pedestrian-proximity-detector/models/mask_rcnn_inception_v2_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix /media/adam/HDD\ Storage/Datasets/supervisely-persons-640x480-tf-records/model.ckpt-3197 --output_directory /home/adam/Development/pedestrian-proximity-detector/models/mask_rcnn_inception_v2_supervisely_640x480_2018_04_10
```

Instructions for installing the Intel RealSense SDK: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
