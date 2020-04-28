#coding:utf-8
#  Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from paddlehub.finetune.task.detection_task import DetectionTask


class YOLOTask(DetectionTask):
    def __init__(self,
                 feature,
                 num_classes,
                 feed_list,
                 data_reader,
                 startup_program=None,
                 config=None,
                 metrics_choices="default"):
        super(YOLOTask, self).__init__(
            data_reader=data_reader,
            num_classes=num_classes,
            feed_list=feed_list,
            feature=feature,
            model_type='yolo',
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)

        self._base_feed_list = feed_list
        self.feature = feature
        self.num_classes = num_classes

    def _parse_anchors(self, anchors):
        """
        Check ANCHORS/ANCHOR_MASKS in config and parse mask_anchors
        """
        self.anchors = []
        self.mask_anchors = []

        assert len(anchors) > 0, "ANCHORS not set."
        assert len(self.anchor_masks) > 0, "ANCHOR_MASKS not set."

        for anchor in anchors:
            assert len(anchor) == 2, "anchor {} len should be 2".format(anchor)
            self.anchors.extend(anchor)

        anchor_num = len(anchors)
        for masks in self.anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def _build_net(self):
        if self.is_predict_phase:
            self.env.labels = self._add_label()
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119],
                   [116, 90], [156, 198], [373, 326]]
        self._parse_anchors(anchors)

        tip_list = self.feature
        outputs = []
        for i, tip in enumerate(tip_list):
            # out channel number = mask_num * (5 + class_num)
            num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                # Rename for: conflict with module pretrain weights
                param_attr=ParamAttr(
                    name="paddlehub_yolo_output.{}.conv.weights".format(i)),
                bias_attr=ParamAttr(
                    regularizer=L2Decay(0.),
                    name="paddlehub_yolo_output.{}.conv.bias".format(i)))
            outputs.append(block_out)

        if self.is_train_phase:
            return outputs

        im_size = self.feed_var_list[1]
        boxes = []
        scores = []
        downsample = 32
        for i, output in enumerate(outputs):
            box, score = fluid.layers.yolo_box(
                x=output,
                img_size=im_size,
                anchors=self.mask_anchors[i],
                class_num=self.num_classes,
                conf_thresh=0.01,
                downsample_ratio=downsample,
                name="yolo_box" + str(i))
            boxes.append(box)
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
            downsample //= 2
        yolo_boxes = fluid.layers.concat(boxes, axis=1)
        yolo_scores = fluid.layers.concat(scores, axis=2)
        # pred = self.nms(bboxes=yolo_boxes, scores=yolo_scores)
        pred = fluid.layers.multiclass_nms(
            bboxes=yolo_boxes,
            scores=yolo_scores,
            score_threshold=.01,
            nms_top_k=1000,
            keep_top_k=100,
            nms_threshold=0.45,
            normalized=False,
            nms_eta=1.0,
            background_label=-1)
        return [pred]

    def _add_label(self):
        if self.is_train_phase:
            idx_list = [1, 2, 3]  # 'gt_box', 'gt_label', 'gt_score'
        elif self.is_test_phase:
            idx_list = [2, 3, 4,
                        5]  # 'im_id', 'gt_box', 'gt_label', 'is_difficult'
        else:  # predict
            idx_list = [2]
        return self._add_label_by_fields(idx_list)

    def _add_loss(self):
        if self.is_train_phase:
            gt_box, gt_label, gt_score = self.labels
            outputs = self.outputs
            losses = []
            downsample = 32
            for i, output in enumerate(outputs):
                anchor_mask = self.anchor_masks[i]
                loss = fluid.layers.yolov3_loss(
                    x=output,
                    gt_box=gt_box,
                    gt_label=gt_label,
                    gt_score=gt_score,
                    anchors=self.anchors,
                    anchor_mask=anchor_mask,
                    class_num=self.num_classes,
                    ignore_thresh=0.7,
                    downsample_ratio=downsample,
                    use_label_smooth=True,
                    name="yolo_loss" + str(i))
                losses.append(fluid.layers.reduce_mean(loss))
                downsample //= 2

            loss = sum(losses)
        else:
            loss = fluid.layers.fill_constant(
                shape=[1], value=-1, dtype='float32')
        return loss

    def _feed_list(self, for_export=False):
        feed_list = [varname for varname in self._base_feed_list]
        if self.is_train_phase:
            return [feed_list[0]] + [label.name for label in self.labels]
        elif self.is_test_phase:
            return feed_list + [label.name for label in self.labels]
        if for_export:
            return feed_list[:2]
        else:
            return feed_list + [self.labels[0].name]

    def _fetch_list(self, for_export=False):
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        if self.is_train_phase:
            return [self.loss.name]
        elif self.is_test_phase:
            # im_shape, im_id, bbox
            return [
                self.feed_list[1], self.labels[0].name, self.outputs[0].name,
                self.loss.name
            ]

        # im_shape, im_id, bbox
        if for_export:
            return [self.outputs[0].name]
        else:
            return [
                self.feed_list[1], self.labels[0].name, self.outputs[0].name
            ]
