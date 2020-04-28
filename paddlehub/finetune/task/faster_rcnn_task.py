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
from paddle.fluid.initializer import Normal

from paddlehub.common.paddle_helper import clone_program
from paddlehub.finetune.task.detection_task import DetectionTask


class FasterRCNNTask(DetectionTask):
    def __init__(self,
                 data_reader,
                 num_classes,
                 feed_list,
                 feature,
                 predict_feed_list=None,
                 predict_feature=None,
                 startup_program=None,
                 config=None,
                 metrics_choices="default"):
        super(FasterRCNNTask, self).__init__(
            data_reader=data_reader,
            num_classes=num_classes,
            feed_list=feed_list,
            feature=feature,
            model_type='rcnn',
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)

        self._base_feed_list = feed_list
        self._base_predict_feed_list = predict_feed_list
        self.feature = feature
        self.predict_feature = predict_feature
        self.num_classes = num_classes
        if predict_feature:
            self._base_predict_main_program = clone_program(
                predict_feature[0].block.program, for_test=False)
        else:
            self._base_predict_main_program = None

    def _build_net(self):
        if self.is_train_phase:
            head_feat = self.feature[0]
        else:
            if self.is_predict_phase:
                self.env.labels = self._add_label()
            head_feat = self.main_program.global_block().vars[
                self.predict_feature[0].name]

        cls_score = fluid.layers.fc(
            input=head_feat,
            size=self.num_classes,
            act=None,
            name='paddlehub_rcnn_cls_score',
            param_attr=ParamAttr(
                name='paddlehub_rcnn_cls_score_weights',
                initializer=Normal(loc=0.0, scale=0.01)),
            bias_attr=ParamAttr(
                name='paddlehub_rcnn_cls_score_bias',
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        bbox_pred = fluid.layers.fc(
            input=head_feat,
            size=4 * self.num_classes,
            act=None,
            name='paddlehub_rcnn_bbox_pred',
            param_attr=ParamAttr(
                name='paddlehub_rcnn_bbox_pred_weights',
                initializer=Normal(loc=0.0, scale=0.001)),
            bias_attr=ParamAttr(
                name='paddlehub_rcnn_bbox_pred_bias',
                learning_rate=2.,
                regularizer=L2Decay(0.)))

        if self.is_train_phase:
            rpn_cls_loss, rpn_reg_loss, outs = self.feature[1:]
            labels_int32 = outs[1]
            bbox_targets = outs[2]
            bbox_inside_weights = outs[3]
            bbox_outside_weights = outs[4]
            labels_int64 = fluid.layers.cast(x=labels_int32, dtype='int64')
            labels_int64.stop_gradient = True
            loss_cls = fluid.layers.softmax_with_cross_entropy(
                logits=cls_score, label=labels_int64, numeric_stable_mode=True)
            loss_cls = fluid.layers.reduce_mean(loss_cls)
            loss_bbox = fluid.layers.smooth_l1(
                x=bbox_pred,
                y=bbox_targets,
                inside_weight=bbox_inside_weights,
                outside_weight=bbox_outside_weights,
                sigma=1.0)
            loss_bbox = fluid.layers.reduce_mean(loss_bbox)
            total_loss = fluid.layers.sum(
                [loss_bbox, loss_cls, rpn_cls_loss, rpn_reg_loss])
            return [total_loss]
        else:
            rois = self.main_program.global_block().vars[
                self.predict_feature[1].name]
            im_info = self.feed_var_list[1]
            im_shape = self.feed_var_list[3]
            im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
            im_scale = fluid.layers.sequence_expand(im_scale, rois)
            boxes = rois / im_scale
            cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
            bbox_pred = fluid.layers.reshape(bbox_pred,
                                             (-1, self.num_classes, 4))
            # decoded_box = self.box_coder(prior_box=boxes, target_box=bbox_pred)
            decoded_box = fluid.layers.box_coder(
                prior_box=boxes,
                prior_box_var=[0.1, 0.1, 0.2, 0.2],
                target_box=bbox_pred,
                code_type='decode_center_size',
                box_normalized=False,
                axis=1)
            cliped_box = fluid.layers.box_clip(
                input=decoded_box, im_info=im_shape)
            # pred_result = self.nms(bboxes=cliped_box, scores=cls_prob)
            pred_result = fluid.layers.multiclass_nms(
                bboxes=decoded_box,
                scores=cls_prob,
                score_threshold=.05,
                nms_top_k=-1,
                keep_top_k=100,
                nms_threshold=.5,
                normalized=False,
                nms_eta=1.0,
                background_label=0)
            return [pred_result]

    def _add_label(self):
        if self.is_train_phase:
            # 'im_id'
            idx_list = [2]
        elif self.is_test_phase:
            # 'im_id', 'gt_box', 'gt_label', 'is_difficult'
            idx_list = [2, 4, 5, 6]
        else:  # predict
            idx_list = [2]
        return self._add_label_by_fields(idx_list)

    def _add_loss(self):
        if self.is_train_phase:
            loss = self.env.outputs[-1]
        else:
            loss = fluid.layers.fill_constant(
                shape=[1], value=-1, dtype='float32')
        return loss

    def _feed_list(self, for_export=False):
        if self.is_train_phase:
            feed_list = [varname for varname in self._base_feed_list]
        else:
            feed_list = [varname for varname in self._base_predict_feed_list]

        if self.is_train_phase:
            # feed_list is ['image', 'im_info', 'gt_box', 'gt_label', 'is_crowd']
            return feed_list[:2] + [self.labels[0].name] + feed_list[2:]
        elif self.is_test_phase:
            # feed list is ['image', 'im_info', 'im_shape']
            return feed_list[:2] + [self.labels[0].name] + feed_list[2:] + \
                   [label.name for label in self.labels[1:]]
        if for_export:
            # skip im_id
            return feed_list[:2] + feed_list[3:]
        else:
            return feed_list[:2] + [self.labels[0].name] + feed_list[2:]

    def _fetch_list(self, for_export=False):
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        if self.is_train_phase:
            return [self.loss.name]
        elif self.is_test_phase:
            # im_shape, im_id, bbox
            return [
                self.feed_list[2], self.labels[0].name, self.outputs[0].name,
                self.loss.name
            ]

        # im_shape, im_id, bbox
        if for_export:
            return [self.outputs[0].name]
        else:
            return [
                self.feed_list[2], self.labels[0].name, self.outputs[0].name
            ]

    @property
    def base_main_program(self):
        if self.is_train_phase:
            return self._base_main_program
        return self._base_predict_main_program
