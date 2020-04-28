# -*- coding:utf8 -*-
import argparse
import os
import ast

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.reader.cv_reader import ObjectDetectionReader
from paddlehub.dataset.base_cv_dataset import ObjectDetectionDataset

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--use_gpu",            type=ast.literal_eval,  default=True,                                help="Whether use GPU for fine-tuning.")
parser.add_argument("--checkpoint_dir",     type=str,               default="faster_rcnn_finetune_ckpt",         help="Path to save log data.")
parser.add_argument("--batch_size",         type=int,               default=8,                                   help="Total examples' number in batch for training.")
parser.add_argument("--module",             type=str,               default="faster_rcnn_resnet50_coco2017",     help="Module used as feature extractor.")
parser.add_argument("--dataset",            type=str,               default="coco_10",                           help="Dataset to finetune.")
# yapf: enable.


def predict(args):
    module = hub.Module(name=args.module)
    dataset = hub.dataset.Coco10('rcnn')

    print("dataset.num_labels:", dataset.num_labels)

    # define batch reader
    data_reader = ObjectDetectionReader(dataset=dataset, model_type='rcnn')
    pred_input_dict, pred_output_dict, pred_program = module.context(
        trainable=False, phase='predict')

    pred_feed_list = [
        pred_input_dict['image'].name, pred_input_dict['im_info'].name,
        pred_input_dict['im_shape'].name
    ]

    pred_feature = [pred_output_dict['head_feat'], pred_output_dict['rois']]

    config = hub.RunConfig(
        use_data_parallel=False,
        use_pyreader=True,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    task = hub.FasterRCNNTask(
        data_reader=data_reader,
        num_classes=dataset.num_labels,
        predict_feed_list=pred_feed_list,
        predict_feature=pred_feature,
        config=config)

    data = [
        "./test/test_img_bird.jpg",
        "./test/test_img_cat.jpg",
    ]
    label_map = dataset.label_dict()
    results = task.predict(data=data, return_result=True, accelerate_mode=False)
    print(results)


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args)
