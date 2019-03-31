# pedestrian-proximity-detector

Download links for pretrained Tensorflow Object Detection API models can be found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Training command:

```
python train.py --logtostderr --train_dir=/media/adam/HDD\ Storage/Datasets/supervisely-persons-tf-records ^ --pipeline_config_path=/home/adam/Development/pedestrian-proximity-detector/models/mask_rcnn_inception_v2_coco_2018_01_28/pipeline.config
```
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/adam/Development/pedestrian-proximity-detector/models/mask_rcnn_inception_v2_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix /media/adam/HDD\ Storage/Datasets/supervisely-persons-tf-records/model.ckpt-3197 --output_directory /home/adam/Development/pedestrian-proximity-detector/models/mask_rcnn_inception_v2_supervisely_2018_03_31
```
