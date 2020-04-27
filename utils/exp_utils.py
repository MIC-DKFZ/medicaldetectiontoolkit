#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
# ==============================================================================
from typing import Iterable, Tuple, Any
import sys
import subprocess
from multiprocessing import Process
import os

import plotting
import importlib.util
import pickle

import logging
from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict
import numpy as np
import torch
import pandas as pd

def split_off_process(target, *args, daemon=False, **kwargs):
    """Start a process that won't block parent script.
    No join(), no return value. If daemon=False: before parent exits, it waits for this to finish.
    """
    p = Process(target=target, args=tuple(args), kwargs=kwargs, daemon=daemon)
    p.start()
    return p

class CombinedLogger(object):
    """Combine console and tensorboard logger and record system metrics.
    """

    def __init__(self, name, log_dir, server_env=True, fold="all"):
        self.pylogger = logging.getLogger(name)
        self.tboard = SummaryWriter(log_dir=os.path.join(log_dir, "tboard"))
        self.log_dir = log_dir
        self.fold = str(fold)
        self.server_env = server_env

        self.pylogger.setLevel(logging.DEBUG)
        self.log_file = os.path.join(log_dir, "fold_"+self.fold, 'exec.log')
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.pylogger.addHandler(logging.FileHandler(self.log_file))
        if not server_env:
            self.pylogger.addHandler(ColorHandler())
        else:
            self.pylogger.addHandler(logging.StreamHandler())
        self.pylogger.propagate = False

    def __getattr__(self, attr):
        """delegate all undefined method requests to objects of
        this class in order pylogger, tboard (first find first serve).
        E.g., combinedlogger.add_scalars(...) should trigger self.tboard.add_scalars(...)
        """
        for obj in [self.pylogger, self.tboard]:
            if attr in dir(obj):
                return getattr(obj, attr)
        print("logger attr not found")

    def set_logfile(self, fold=None, log_file=None):
        if fold is not None:
            self.fold = str(fold)
        if log_file is None:
            self.log_file = os.path.join(self.log_dir, "fold_"+self.fold, 'exec.log')
        else:
            self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        for hdlr in self.pylogger.handlers:
            hdlr.close()
        self.pylogger.handlers = []
        self.pylogger.addHandler(logging.FileHandler(self.log_file))
        if not self.server_env:
            self.pylogger.addHandler(ColorHandler())
        else:
            self.pylogger.addHandler(logging.StreamHandler())

    def metrics2tboard(self, metrics, global_step=None, suptitle=None):
        """
        :param metrics: {'train': dataframe, 'val':df}, df as produced in
            evaluator.py.evaluate_predictions
        """
        # print("metrics", metrics)
        if global_step is None:
            global_step = len(metrics['train'][list(metrics['train'].keys())[0]]) - 1
        if suptitle is not None:
            suptitle = str(suptitle)
        else:
            suptitle = "Fold_" + str(self.fold)

        for key in ['train', 'val']:
            # series = {k:np.array(v[-1]) for (k,v) in metrics[key].items() if not np.isnan(v[-1]) and not 'Bin_Stats' in k}
            loss_series = {}
            mon_met_series = {}
            for tag, val in metrics[key].items():
                val = val[-1]  # maybe remove list wrapping, recording in evaluator?
                if 'loss' in tag.lower() and not np.isnan(val):
                    loss_series["{}".format(tag)] = val
                elif not np.isnan(val):
                    mon_met_series["{}".format(tag)] = val

            self.tboard.add_scalars(suptitle + "/Losses/{}".format(key), loss_series, global_step)
            self.tboard.add_scalars(suptitle + "/Monitor_Metrics/{}".format(key), mon_met_series, global_step)
        self.tboard.add_scalars(suptitle + "/Learning_Rate", metrics["lr"], global_step)
        return

    def __del__(self):  # otherwise might produce multiple prints e.g. in ipython console
        for hdlr in self.pylogger.handlers:
            hdlr.close()
        self.pylogger.handlers = []
        del self.pylogger
        self.tboard.flush()
        # close somehow prevents main script from exiting
        # maybe revise this issue in a later pytorch version
        #self.tboard.close()


def get_logger(exp_dir, server_env=False):
    """
    creates logger instance. writing out info to file, to terminal and to tensorboard.
    :param exp_dir: experiment directory, where exec.log file is stored.
    :param server_env: True if operating in server environment (e.g., gpu cluster)
    :return: custom CombinedLogger instance.
    """
    log_dir = os.path.join(exp_dir, "logs")
    logger = CombinedLogger('medicaldetectiontoolkit', log_dir, server_env=server_env)
    print("Logging to {}".format(logger.log_file))
    return logger


def prep_exp(dataset_path, exp_path, server_env, use_stored_settings=True, is_training=True):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/inference of this experiment can be started at anytime. Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param server_env: boolean flag. pass to configs script for cloud deployment.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :param is_training: boolean flag. distinguishes train vs. inference mode.
    :return:
    """

    if is_training:
        if use_stored_settings:
            cf_file = import_module('cf_file', os.path.join(exp_path, 'configs.py'))
            cf = cf_file.configs(server_env)
            # in this mode, previously saved model and backbone need to be found in exp dir.
            if not os.path.isfile(os.path.join(exp_path, 'model.py')) or \
                    not os.path.isfile(os.path.join(exp_path, 'backbone.py')):
                raise Exception(
                    "Selected use_stored_settings option but no model and/or backbone source files exist in exp dir.")
            cf.model_path = os.path.join(exp_path, 'model.py')
            cf.backbone_path = os.path.join(exp_path, 'backbone.py')
        else:
            # this case overwrites settings files in exp dir, i.e., default_configs, configs, backbone, model
            os.makedirs(exp_path, exist_ok=True)
            # run training with source code info and copy snapshot of model to exp_dir for later testing (overwrite scripts if exp_dir already exists.)
            subprocess.call('cp {} {}'.format('default_configs.py', os.path.join(exp_path, 'default_configs.py')),
                            shell=True)
            subprocess.call(
                'cp {} {}'.format(os.path.join(dataset_path, 'configs.py'), os.path.join(exp_path, 'configs.py')),
                shell=True)
            cf_file = import_module('cf_file', os.path.join(dataset_path, 'configs.py'))
            cf = cf_file.configs(server_env)
            subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(exp_path, 'model.py')), shell=True)
            subprocess.call('cp {} {}'.format(cf.backbone_path, os.path.join(exp_path, 'backbone.py')), shell=True)
            if os.path.isfile(os.path.join(exp_path, "fold_ids.pickle")):
                subprocess.call('rm {}'.format(os.path.join(exp_path, "fold_ids.pickle")), shell=True)

    else:
        # testing, use model and backbone stored in exp dir.
        cf_file = import_module('cf_file', os.path.join(exp_path, 'configs.py'))
        cf = cf_file.configs(server_env)
        cf.model_path = os.path.join(exp_path, 'model.py')
        cf.backbone_path = os.path.join(exp_path, 'backbone.py')


    cf.exp_dir = exp_path
    cf.test_dir = os.path.join(cf.exp_dir, 'test')
    cf.plot_dir = os.path.join(cf.exp_dir, 'plots')
    if not os.path.exists(cf.test_dir):
        os.mkdir(cf.test_dir)
    if not os.path.exists(cf.plot_dir):
        os.mkdir(cf.plot_dir)
    cf.experiment_name = exp_path.split("/")[-1]
    cf.created_fold_id_pickle = False

    return cf



def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_params_flag(module: torch.nn.Module, flag: Tuple[str, Any], check_overwrite: bool = True):
    """Set an attribute for all module parameters.

    :param flag: tuple (str attribute name : attr value)
    :param check_overwrite: if True, assert that attribute not already exists.

    """
    for param in module.parameters():
        if check_overwrite:
            assert not hasattr(param, flag[0]), \
                "param {} already has attr {} (w/ val {})".format(param, flag[0], getattr(param, flag[0]))
        setattr(param, flag[0], flag[1])
    return module

def parse_params_for_optim(net: torch.nn.Module, weight_decay: float = 0., exclude_from_wd: Iterable = ("norm",)):
    """Format network parameters for the optimizer.
    Convenience function to include options for group-specific settings like weight decay.
    :param net:
    :param weight_decay:
    :param exclude_from_wd: List of strings of parameter-group names to exclude from weight decay. Options: "norm", "bias".
    :return:
    """
    # pytorch implements parameter groups as dicts {'params': ...} and
    # weight decay as p.data.mul_(1 - group['lr'] * group['weight_decay'])
    norm_types = [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                  torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
                  torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.SyncBatchNorm, torch.nn.LocalResponseNorm
                  ]
    level_map = {"bias": "weight",
                 "norm": "module"}
    type_map = {"norm": norm_types}

    exclude_from_wd = [str(name).lower() for name in exclude_from_wd]
    exclude_weight_names = [k for k, v in level_map.items() if k in exclude_from_wd and v == "weight"]
    exclude_module_types = tuple([type_ for k, v in level_map.items() if (k in exclude_from_wd and v == "module")
                                  for type_ in type_map[k]])

    if exclude_from_wd:
        print("excluding {} from weight decay.".format(exclude_from_wd))

    for module in net.modules():
        if isinstance(module, exclude_module_types):
            set_params_flag(module, ("no_wd", True))
    for param_name, param in net.named_parameters():
        if np.any([ename in param_name for ename in exclude_weight_names]):
            setattr(param, "no_wd", True)

    with_dec, no_dec = [], []
    for param in net.parameters():
        if hasattr(param, "no_wd") and param.no_wd == True:
            no_dec.append(param)
        else:
            with_dec.append(param)
    orig_ps = sum(p.numel() for p in net.parameters())
    with_ps = sum(p.numel() for p in with_dec)
    wo_ps = sum(p.numel() for p in no_dec)
    assert orig_ps == with_ps + wo_ps, "orig n parameters {} unequals sum of with wd {} and w/o wd {}."\
        .format(orig_ps, with_ps, wo_ps)

    groups = [{'params': gr, 'weight_decay': wd} for (gr, wd) in [(no_dec, 0.), (with_dec, weight_decay)] if len(gr)>0]
    return groups


class ModelSelector:
    '''
    saves a checkpoint after each epoch as 'last_state' (can be loaded to continue interrupted training).
    saves the top-k (k=cf.save_n_models) ranked epochs. In inference, predictions of multiple epochs can be ensembled to improve performance.
    '''

    def __init__(self, cf, logger):

        self.cf = cf
        self.saved_epochs = [-1] * cf.save_n_models
        self.logger = logger

    def run_model_selection(self, net, optimizer, monitor_metrics, epoch):

        # take the mean over all selection criteria in each epoch
        non_nan_scores = np.mean(np.array([[0 if (ii is None or np.isnan(ii)) else ii for ii in monitor_metrics['val'][sc]] for sc in self.cf.model_selection_criteria]), 0)
        epochs_scores = [ii for ii in non_nan_scores[1:]]
        # ranking of epochs according to model_selection_criterion
        epoch_ranking = np.argsort(epochs_scores, kind="stable")[::-1] + 1 #epochs start at 1
        # if set in configs, epochs < min_save_thresh are discarded from saving process.
        epoch_ranking = epoch_ranking[epoch_ranking >= self.cf.min_save_thresh]

        # check if current epoch is among the top-k epochs.
        if epoch in epoch_ranking[:self.cf.save_n_models]:

            save_dir = os.path.join(self.cf.fold_dir, '{}_best_checkpoint'.format(epoch))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            torch.save(net.state_dict(), os.path.join(save_dir, 'params.pth'))
            with open(os.path.join(save_dir, 'monitor_metrics.pickle'), 'wb') as handle:
                pickle.dump(monitor_metrics, handle)
            # save epoch_ranking to keep info for inference.
            np.save(os.path.join(self.cf.fold_dir, 'epoch_ranking'), epoch_ranking[:self.cf.save_n_models])
            np.save(os.path.join(save_dir, 'epoch_ranking'), epoch_ranking[:self.cf.save_n_models])

            self.logger.info(
                "saving current epoch {} at rank {}".format(epoch, np.argwhere(epoch_ranking == epoch)))
            # delete params of the epoch that just fell out of the top-k epochs.
            for se in [int(ii.split('_')[0]) for ii in os.listdir(self.cf.fold_dir) if 'best_checkpoint' in ii]:
                if se in epoch_ranking[self.cf.save_n_models:]:
                    subprocess.call('rm -rf {}'.format(os.path.join(self.cf.fold_dir, '{}_best_checkpoint'.format(se))), shell=True)
                    self.logger.info('deleting epoch {} at rank {}'.format(se, np.argwhere(epoch_ranking == se)))

        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint of current epoch.
        save_dir = os.path.join(self.cf.fold_dir, 'last_checkpoint'.format(epoch))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(state, os.path.join(save_dir, 'params.pth'))
        np.save(os.path.join(save_dir, 'epoch_ranking'), epoch_ranking[:self.cf.save_n_models])
        with open(os.path.join(save_dir, 'monitor_metrics.pickle'), 'wb') as handle:
            pickle.dump(monitor_metrics, handle)



def load_checkpoint(checkpoint_path, net, optimizer):

    checkpoint = torch.load(os.path.join(checkpoint_path, 'params.pth'))
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    with open(os.path.join(checkpoint_path, 'monitor_metrics.pickle'), 'rb') as handle:
        monitor_metrics = pickle.load(handle)
    starting_epoch = checkpoint['epoch'] + 1
    return starting_epoch, net, optimizer, monitor_metrics



def prepare_monitoring(cf):
    """
    creates dictionaries, where train/val metrics are stored.
    """
    metrics = {}
    # first entry for loss dict accounts for epoch starting at 1.
    metrics['train'] = OrderedDict()
    metrics['val'] = OrderedDict()
    metric_classes = []
    if 'rois' in cf.report_score_level:
        metric_classes.extend([v for k, v in cf.class_dict.items()])
    if 'patient' in cf.report_score_level:
        metric_classes.extend(['patient'])
    for cl in metric_classes:
        metrics['train'][cl + '_ap'] = [np.nan]
        metrics['val'][cl + '_ap'] = [np.nan]
        if cl == 'patient':
            metrics['train'][cl + '_auc'] = [np.nan]
            metrics['val'][cl + '_auc'] = [np.nan]

    return metrics



def create_csv_output(results_list, cf, logger):
    """
    Write out test set predictions to .csv file. output format is one line per prediction:
    PatientID | PredictionID | [y1 x1 y2 x2 (z1) (z2)] | score | pred_classID
    Note, that prediction coordinates correspond to images as loaded for training/testing and need to be adapted when
    plotted over raw data (before preprocessing/resampling).
    :param results_list: [[patient_results, patient_id], [patient_results, patient_id], ...]
    """

    logger.info('creating csv output file at {}'.format(os.path.join(cf.test_dir, 'results.csv')))
    predictions_df = pd.DataFrame(columns = ['patientID', 'predictionID', 'coords', 'score', 'pred_classID'])
    for r in results_list:

        pid = r[1]

        #optionally load resampling info from preprocessing to match output predictions with raw data.
        #with open(os.path.join(cf.exp_dir, 'test_resampling_info', pid), 'rb') as handle:
        #    resampling_info = pickle.load(handle)

        for bix, box in enumerate(r[0][0]):
            if box["box_type"] == "gt":
                continue
            assert box['box_type'] == 'det', box['box_type']
            coords = box['box_coords']
            score = box['box_score']
            pred_class_id = box['box_pred_class_id']
            out_coords = []
            if score >= cf.min_det_thresh:
                out_coords.append(coords[0]) #* resampling_info['scale'][0])
                out_coords.append(coords[1]) #* resampling_info['scale'][1])
                out_coords.append(coords[2]) #* resampling_info['scale'][0])
                out_coords.append(coords[3]) #* resampling_info['scale'][1])
                if len(coords) > 4:
                    out_coords.append(coords[4]) #* resampling_info['scale'][2] + resampling_info['z_crop'])
                    out_coords.append(coords[5]) #* resampling_info['scale'][2] + resampling_info['z_crop'])

                predictions_df.loc[len(predictions_df)] = [pid, bix, out_coords, score, pred_class_id]
    try:
        fold = cf.fold
    except:
        fold = 'hold_out'
    predictions_df.to_csv(os.path.join(cf.exp_dir, 'results_{}.csv'.format(fold)), index=False)



class _AnsiColorizer(object):
    """
    A colorizer is an object that loosely wraps around a stream, allowing
    callers to write text to the stream in a particular color.

    Colorizer classes must implement C{supported()} and C{write(text, color)}.
    """
    _colors = dict(black=30, red=31, green=32, yellow=33,
                   blue=34, magenta=35, cyan=36, white=37, default=39)

    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def supported(cls, stream=sys.stdout):
        """
        A class method that returns True if the current platform supports
        coloring terminal output using this method. Returns False otherwise.
        """
        if not stream.isatty():
            return False  # auto color only on TTYs
        try:
            import curses
        except ImportError:
            return False
        else:
            try:
                try:
                    return curses.tigetnum("colors") > 2
                except curses.error:
                    curses.setupterm()
                    return curses.tigetnum("colors") > 2
            except:
                raise
                # guess false in case of error
                return False

    def write(self, text, color):
        """
        Write the given text to the stream in the given color.

        @param text: Text to be written to the stream.

        @param color: A string label for a color. e.g. 'red', 'white'.
        """
        color = self._colors[color]
        self.stream.write('\x1b[%sm%s\x1b[0m' % (color, text))



class ColorHandler(logging.StreamHandler):


    def __init__(self, stream=sys.stdout):
        super(ColorHandler, self).__init__(_AnsiColorizer(stream))

    def emit(self, record):
        msg_colors = {
            logging.DEBUG: "green",
            logging.INFO: "default",
            logging.WARNING: "red",
            logging.ERROR: "red"
        }
        color = msg_colors.get(record.levelno, "blue")
        self.stream.write(record.msg + "\n", color)

