import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="~/.torch/models/_detectron_35858933_12_2017_baselines_e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC_output_train_coco_2014_train%3Acoco_2014_valminusminival_generalized_rcnn_model_final.pkl",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="../models/mask_rcnn_resnet_50_supervisely/mask_rcnn_R-50-FPN_1x_detectron_no_last_layers.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="../../maskrcnn-benchmark/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d

keys_to_remove = ['cls_score.bias', 'cls_score.weight', 'bbox_pred.bias', 'bbox_pred.weight',
                    'mask_fcn4.bias', 'mask_fcn4.weight']
                    # 'mask_fcn3.bias', 'mask_fcn3.weight',
                    # 'mask_fcn2.bias', 'mask_fcn2.weight',
                    # 'mask_fcn1.bias', 'mask_fcn1.weight',]

newdict['model'] = removekey(_d['model'], keys_to_remove)

newdict['model'] = {k.replace('module.', ''): v for k, v in newdict['model'].items()
                                    if 'cls_score' not in k and 'bbox_pred' not in k and 'mask_fcn_logits' not in k}
torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))
