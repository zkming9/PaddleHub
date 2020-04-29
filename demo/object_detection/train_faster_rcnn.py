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
parser.add_argument("--num_epoch",          type=int,               default=50,                               help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu",            type=ast.literal_eval,  default=True,                             help="Whether use GPU for fine-tuning.")
parser.add_argument("--checkpoint_dir",     type=str,               default="faster_rcnn_finetune_ckpt",      help="Path to save log data.")
parser.add_argument("--batch_size",         type=int,               default=1,                                help="Total examples' number in batch for training.")
parser.add_argument("--module",             type=str,               default="faster_rcnn_resnet50_coco2017",  help="Module used as feature extractor.")
parser.add_argument("--use_data_parallel",  type=ast.literal_eval,  default=False,                            help="Whether use data parallel.")
# yapf: enable.


def finetune(args):
    module = hub.Module(name=args.module)
    dataset = hub.dataset.Balloon('rcnn')

    print("dataset.num_labels:", dataset.num_labels)

    # define batch reader
    data_reader = ObjectDetectionReader(dataset=dataset, model_type='rcnn')

    input_dict, output_dict, program = module.context(
        trainable=True, num_classes=dataset.num_labels)
    pred_input_dict, pred_output_dict, pred_program = module.context(
        trainable=False, phase='predict', num_classes=dataset.num_labels)

    feed_list = [
        input_dict["image"].name, input_dict["im_info"].name,
        input_dict['gt_bbox'].name, input_dict['gt_class'].name,
        input_dict['is_crowd'].name
    ]

    pred_feed_list = [
        pred_input_dict['image'].name, pred_input_dict['im_info'].name,
        pred_input_dict['im_shape'].name
    ]

    feature = [
        output_dict['head_features'], output_dict['rpn_cls_loss'],
        output_dict['rpn_reg_loss'], output_dict['generate_proposal_labels']
    ]

    pred_feature = [pred_output_dict['head_features'], pred_output_dict['rois']]

    config = hub.RunConfig(
        log_interval=10,
        eval_interval=10,
        use_data_parallel=args.use_data_parallel,
        use_pyreader=True,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy(
            learning_rate=0.00025, optimizer_name="momentum", momentum=0.9))

    task = hub.FasterRCNNTask(
        data_reader=data_reader,
        num_classes=dataset.num_labels,
        feed_list=feed_list,
        feature=feature,
        predict_feed_list=pred_feed_list,
        predict_feature=pred_feature,
        config=config)
    task.finetune_and_eval()


if __name__ == "__main__":
    args = parser.parse_args()
    finetune(args)
