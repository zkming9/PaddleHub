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

import time
from collections import OrderedDict
import numpy as np
import six
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Xavier
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA

from paddlehub.finetune.task.base_task import BaseTask
from paddlehub.contrib.ppdet.utils.eval_utils import eval_results
from paddlehub.contrib.ppdet.utils.coco_eval import bbox2out
from paddlehub.common import detection_config as dconf
from paddlehub.common.paddle_helper import clone_program

feed_var_def = [
    {
        'name': 'im_info',
        'shape': [3],
        'dtype': 'float32',
        'lod_level': 0
    },
    {
        'name': 'im_id',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 0
    },
    {
        'name': 'gt_box',
        'shape': [4],
        'dtype': 'float32',
        'lod_level': 1
    },
    {
        'name': 'gt_label',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 1
    },
    {
        'name': 'is_crowd',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 1
    },
    {
        'name': 'gt_mask',
        'shape': [2],
        'dtype': 'float32',
        'lod_level': 3
    },
    {
        'name': 'is_difficult',
        'shape': [1],
        'dtype': 'int32',
        'lod_level': 1
    },
    {
        'name': 'gt_score',
        'shape': [1],
        'dtype': 'float32',
        'lod_level': 0
    },
    {
        'name': 'im_shape',
        'shape': [3],
        'dtype': 'float32',
        'lod_level': 0
    },
    {
        'name': 'im_size',
        'shape': [2],
        'dtype': 'int32',
        'lod_level': 0
    },
]


class Feed(object):
    def __init__(self):
        self.dataset = None
        self.with_background = True


class DetectionTask(BaseTask):
    def __init__(self,
                 data_reader,
                 num_classes,
                 feed_list,
                 feature,
                 model_type,
                 startup_program=None,
                 config=None,
                 metrics_choices="default"):
        if metrics_choices == "default":
            metrics_choices = ["ap"]

        main_program = feature[0].block.program
        super(DetectionTask, self).__init__(
            data_reader=data_reader,
            main_program=main_program,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)

        self.model_type = model_type

    def save_inference_model(self,
                             dirname,
                             model_filename=None,
                             params_filename=None):
        with self.phase_guard("predict"):
            fluid.io.save_inference_model(
                dirname=dirname,
                executor=self.exe,
                feeded_var_names=self._feed_list(for_export=True),
                target_vars=self.fetch_var_list,
                main_program=self.main_program,
                model_filename=model_filename,
                params_filename=params_filename)

    @property
    def return_numpy(self):
        return 2  # return lod tensor

    def _add_label_by_fields(self, idx_list):
        feed_var_map = {var['name']: var for var in feed_var_def}
        # tensor padding with 0 is used instead of LoD tensor when
        # num_max_boxes is set
        num_max_boxes = dconf.conf[self.model_type].get('num_max_boxes', None)
        if num_max_boxes is not None:
            feed_var_map['gt_label']['shape'] = [num_max_boxes]
            feed_var_map['gt_score']['shape'] = [num_max_boxes]
            feed_var_map['gt_box']['shape'] = [num_max_boxes, 4]
            feed_var_map['is_difficult']['shape'] = [num_max_boxes]
            feed_var_map['gt_label']['lod_level'] = 0
            feed_var_map['gt_score']['lod_level'] = 0
            feed_var_map['gt_box']['lod_level'] = 0
            feed_var_map['is_difficult']['lod_level'] = 0

        if self.is_train_phase:
            fields = dconf.feed_config[self.model_type]['train']['fields']
        elif self.is_test_phase:
            fields = dconf.feed_config[self.model_type]['dev']['fields']
        else:  # Cannot go to here
            # raise RuntimeError("Cannot go to _add_label in predict phase")
            fields = dconf.feed_config[self.model_type]['predict']['fields']

        labels = []
        for i in idx_list:
            key = fields[i]
            l = fluid.layers.data(
                name=feed_var_map[key]['name'],
                shape=feed_var_map[key]['shape'],
                dtype=feed_var_map[key]['dtype'],
                lod_level=feed_var_map[key]['lod_level'])
            labels.append(l)
        return labels

    def _calculate_metrics(self, run_states):
        loss_sum = run_examples = 0
        run_step = run_time_used = 0
        for run_state in run_states:
            run_examples += run_state.run_examples
            run_step += run_state.run_step
            loss_sum += np.mean(np.array(
                run_state.run_results[-1])) * run_state.run_examples

        run_time_used = time.time() - run_states[0].run_time_begin
        avg_loss = loss_sum / run_examples
        run_speed = run_step / run_time_used

        # The first key will be used as main metrics to update the best model
        scores = OrderedDict()
        if self.is_train_phase:
            return scores, avg_loss, run_speed

        keys = ['im_shape', 'im_id', 'bbox']
        results = []
        for run_state in run_states:
            outs = [
                run_state.run_results[0], run_state.run_results[1],
                run_state.run_results[2]
            ]
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)
            }
            results.append(res)

        is_bbox_normalized = dconf.conf[self.model_type]['is_bbox_normalized']
        eval_feed = Feed()
        eval_feed.with_background = dconf.conf[
            self.model_type]['with_background']
        eval_feed.dataset = self.reader

        for metric in self.metrics_choices:
            if metric == "ap":
                box_ap_stats = eval_results(
                    results, eval_feed, 'COCO', self.num_classes, None,
                    is_bbox_normalized, self.config.checkpoint_dir)
                print("box_ap_stats", box_ap_stats)
                scores["ap"] = box_ap_stats[0]
            else:
                raise ValueError("Not Support Metric: \"%s\"" % metric)

        return scores, avg_loss, run_speed

    def _postprocessing(self, run_states):
        """
        postprocessing the run result, get readable result.

        Args:
            run_states (RunState): the raw run result to be processed

        Returns:
            list: readable result
        """
        results = [run_state.run_results for run_state in run_states]
        for outs in results:
            keys = ['im_shape', 'im_id', 'bbox']
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)
            }
            is_bbox_normalized = dconf.conf[
                self.model_type]['is_bbox_normalized']
            clsid2catid = {}
            try:
                items = self._base_data_reader.label_map.items()
            except:
                items = {idx: idx for idx in range(self.num_classes)}.items()
            for k, v in items:
                clsid2catid[v] = k
            bbox_results = bbox2out([res], clsid2catid, is_bbox_normalized)
        return bbox_results

    def _add_metrics(self):
        return []

    @property
    def feed_list(self):
        return self._feed_list(False)

    @property
    def fetch_list(self):
        # ensure fetch loss at last element in train/test phase
        # ensure fetch 'im_shape', 'im_id', 'bbox' at first three elements in test phase
        return self._fetch_list(False)

    @property
    def fetch_var_list(self):
        fetch_list = self._fetch_list(True)
        vars = self.main_program.global_block().vars
        return [vars[varname] for varname in fetch_list]
