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

from paddlehub.finetune.task.detection_task import DetectionTask


class SSDTask(DetectionTask):
    def __init__(self,
                 feature,
                 num_classes,
                 feed_list,
                 data_reader,
                 multi_box_head_config,
                 startup_program=None,
                 config=None,
                 metrics_choices="default"):
        super(SSDTask, self).__init__(
            data_reader=data_reader,
            num_classes=num_classes,
            feed_list=feed_list,
            feature=feature,
            model_type='ssd',
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)

        self._base_feed_list = feed_list
        self.feature = feature
        self.num_classes = num_classes
        self.multi_box_head_config = multi_box_head_config

    def _build_net(self):
        if self.is_predict_phase:  # add im_id
            self.env.labels = self._add_label()

        feature_list = self.feature
        image = self.feed_var_list[0]

        # fix input size according to its module
        mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
            inputs=feature_list,
            image=image,
            num_classes=self.num_classes,
            **self.multi_box_head_config)

        self.env.mid_vars = [mbox_locs, mbox_confs, box, box_var]

        nmsed_out = fluid.layers.detection_output(
            mbox_locs,
            mbox_confs,
            box,
            box_var,
            background_label=0,
            nms_threshold=0.45,
            nms_top_k=400,
            keep_top_k=200,
            score_threshold=0.01,
            nms_eta=1.0)

        return [nmsed_out]

    def _add_label(self):
        if self.is_train_phase:
            # 'gt_box', 'gt_label'
            idx_list = [1, 2]
        elif self.is_test_phase:
            # 'im_id', 'gt_box', 'gt_label', 'is_difficult'
            idx_list = [2, 3, 4, 5]
        else:
            # im_id
            idx_list = [1]
        return self._add_label_by_fields(idx_list)

    def _add_loss(self):
        if self.is_train_phase:
            gt_box = self.labels[0]
            gt_label = self.labels[1]
        else:  # xTodo: update here when using new module
            gt_box = self.labels[1]
            gt_label = self.labels[2]
        mbox_locs, mbox_confs, box, box_var = self.env.mid_vars
        loss = fluid.layers.ssd_loss(
            location=mbox_locs,
            confidence=mbox_confs,
            gt_box=gt_box,
            gt_label=gt_label,
            prior_box=box,
            prior_box_var=box_var)
        loss = fluid.layers.reduce_sum(loss)
        loss.persistable = True
        return loss

    def _feed_list(self, for_export=False):
        # todo: update when using new module
        feed_list = [varname for varname in self._base_feed_list]
        if self.is_train_phase:
            feed_list = feed_list[:1] + [label.name for label in self.labels]
        elif self.is_test_phase:
            feed_list = feed_list + [label.name for label in self.labels]
        else:  # self.is_predict_phase:
            if for_export:
                feed_list = [feed_list[0]]
            else:
                # 'image', 'im_id', 'im_shape'
                feed_list = [feed_list[0], self.labels[0].name, feed_list[1]]
        return feed_list

    def _fetch_list(self, for_export=False):
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        if self.is_train_phase:
            return [self.loss.name]
        elif self.is_test_phase:
            # xTodo: update when using new module
            # im_id, bbox, dets, loss
            return [
                self._base_feed_list[1], self.labels[0].name,
                self.outputs[0].name, self.loss.name
            ]
        # im_shape, im_id, bbox
        if for_export:
            return [self.outputs[0].name]
        else:
            return [
                self._base_feed_list[1], self.labels[0].name,
                self.outputs[0].name
            ]
