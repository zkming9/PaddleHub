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
parser.add_argument("--num_epoch",          type=int,               default=50,                           help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu",            type=ast.literal_eval,  default=True,                         help="Whether use GPU for fine-tuning.")
parser.add_argument("--checkpoint_dir",     type=str,               default="yolo_finetune_ckpt",         help="Path to save log data.")
parser.add_argument("--batch_size",         type=int,               default=8,                            help="Total examples' number in batch for training.")
parser.add_argument("--module",             type=str,               default="yolov3_darknet53_coco2017",  help="Module used as feature extractor.")
parser.add_argument("--dataset",            type=str,               default="coco_10",                    help="Dataset to finetune.")
parser.add_argument("--use_data_parallel",  type=ast.literal_eval,  default=False,                        help="Whether use data parallel.")
# yapf: enable.


def finetune(args):
    module = hub.Module(name=args.module)
    dataset = hub.dataset.Coco10('yolo')

    print("dataset.num_labels:", dataset.num_labels)

    # define batch reader
    data_reader = ObjectDetectionReader(dataset=dataset, model_type='yolo')

    input_dict, output_dict, program = module.context(trainable=True)
    feed_list = [input_dict["image"].name, input_dict["im_size"].name]
    feature = output_dict['head_features']

    config = hub.RunConfig(
        log_interval=10,
        eval_interval=100,
        use_data_parallel=args.use_data_parallel,
        use_pyreader=True,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy(
            learning_rate=0.00025, optimizer_name="momentum", momentum=0.9))

    task = hub.YOLOTask(
        data_reader=data_reader,
        num_classes=dataset.num_labels,
        feed_list=feed_list,
        feature=feature,
        config=config)
    task.finetune_and_eval()


if __name__ == "__main__":
    args = parser.parse_args()
    finetune(args)
