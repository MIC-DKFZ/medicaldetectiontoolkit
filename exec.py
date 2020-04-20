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

"""execution script."""

import argparse
import os, warnings
import time

import torch

import utils.exp_utils as utils
from evaluator import Evaluator
from predictor import Predictor
from plotting import plot_batch_prediction

for msg in ["Attempting to set identical bottom==top results",
            "This figure includes Axes that are not compatible with tight_layout",
            "Data has no positive values, and therefore cannot be log-scaled.",
            ".*invalid value encountered in double_scalars.*",
            ".*Mean of empty slice.*"]:
    warnings.filterwarnings("ignore", msg)


def train(logger):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))

    net = model.net(cf, logger).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    if cf.dynamic_lr_scheduling:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cf.scheduling_mode, factor=cf.lr_decay_factor,
                                                               patience=cf.scheduling_patience)

    model_selector = utils.ModelSelector(cf, logger)
    train_evaluator = Evaluator(cf, logger, mode='train')
    val_evaluator = Evaluator(cf, logger, mode=cf.val_mode)

    starting_epoch = 1

    # prepare monitoring
    monitor_metrics = utils.prepare_monitoring(cf)

    if cf.resume:
        checkpoint_path = os.path.join(cf.fold_dir, "last_checkpoint")
        starting_epoch, net, optimizer, monitor_metrics = \
            utils.load_checkpoint(checkpoint_path, net, optimizer)
        logger.info('resumed from checkpoint {} to epoch {}'.format(checkpoint_path, starting_epoch))


    logger.info('loading dataset and initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, logger)

    for epoch in range(starting_epoch, cf.num_epochs + 1):

        logger.info('starting training epoch {}'.format(epoch))
        start_time = time.time()

        net.train()
        train_results_list = []
        for bix in range(cf.num_train_batches):
            batch = next(batch_gen['train'])
            tic_fw = time.time()
            results_dict = net.train_forward(batch)
            tic_bw = time.time()
            optimizer.zero_grad()
            results_dict['torch_loss'].backward()
            optimizer.step()
            print('\rtr. batch {0}/{1} (ep. {2}) fw {3:.2f}s / bw {4:.2f} s / total {5:.2f} s || '.format(
                bix + 1, cf.num_train_batches, epoch, tic_bw - tic_fw, time.time() - tic_bw,
                time.time() - tic_fw) + results_dict['logger_string'], flush=True, end="")
            train_results_list.append(({k:v for k,v in results_dict.items() if k != "seg_preds"}, batch["pid"]))
        print()

        _, monitor_metrics['train'] = train_evaluator.evaluate_predictions(train_results_list, monitor_metrics['train'])

        logger.info('generating training example plot.')
        plot_batch_prediction(batch, results_dict, cf, outfile=os.path.join(
            cf.plot_dir, 'pred_example_{}_train.png'.format(cf.fold)))

        train_time = time.time() - start_time

        logger.info('starting validation in mode {}.'.format(cf.val_mode))
        with torch.no_grad():
            net.eval()
            if cf.do_validation:
                val_results_list = []
                val_predictor = Predictor(cf, net, logger, mode='val')
                for _ in range(batch_gen['n_val']):
                    batch = next(batch_gen[cf.val_mode])
                    if cf.val_mode == 'val_patient':
                        results_dict = val_predictor.predict_patient(batch)
                    elif cf.val_mode == 'val_sampling':
                        results_dict = net.train_forward(batch, is_validation=True)
                    #val_results_list.append([results_dict['boxes'], batch['pid']])
                    val_results_list.append(({k:v for k,v in results_dict.items() if k != "seg_preds"}, batch["pid"]))

                _, monitor_metrics['val'] = val_evaluator.evaluate_predictions(val_results_list, monitor_metrics['val'])
                model_selector.run_model_selection(net, optimizer, monitor_metrics, epoch)

            # update monitoring and prediction plots
            monitor_metrics.update({"lr":
                                        {str(g): group['lr'] for (g, group) in enumerate(optimizer.param_groups)}})
            logger.metrics2tboard(monitor_metrics, global_step=epoch)

            epoch_time = time.time() - start_time
            logger.info('trained epoch {}: took {:.2f} s ({:.2f} s train / {:.2f} s val)'.format(
                epoch, epoch_time, train_time, epoch_time-train_time))
            batch = next(batch_gen['val_sampling'])
            results_dict = net.train_forward(batch, is_validation=True)
            logger.info('generating validation-sampling example plot.')
            plot_batch_prediction(batch, results_dict, cf, outfile=os.path.join(
                cf.plot_dir, 'pred_example_{}_val.png'.format(cf.fold)))

        # -------------- scheduling -----------------
        if cf.dynamic_lr_scheduling:
            scheduler.step(monitor_metrics["val"][cf.scheduling_criterion][-1])
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cf.learning_rate[epoch-1]

def test(logger):
    """
    perform testing for a given fold (or hold out set). save stats in evaluator.
    """
    logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
    net = model.net(cf, logger).cuda()
    test_predictor = Predictor(cf, net, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    batch_gen = data_loader.get_test_generator(cf, logger)
    test_results_list = test_predictor.predict_test_set(batch_gen, return_results=True)
    test_evaluator.evaluate_predictions(test_results_list)
    test_evaluator.score_test_df()


if __name__ == '__main__':
    stime = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str,  default='train_test',
                        help='one out of: train / test / train_test / analysis / create_exp')
    parser.add_argument('-f','--folds', nargs='+', type=int, default=None,
                        help='None runs over all folds in CV. otherwise specify list of folds.')
    parser.add_argument('--exp_dir', type=str, default='/path/to/experiment/directory',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--server_env', default=False, action='store_true',
                        help='change IO settings to deploy models on a cluster.')
    parser.add_argument('--data_dest', type=str, default=None, help="path to final data folder if different from config.")
    parser.add_argument('--use_stored_settings', default=False, action='store_true',
                        help='load configs from existing exp_dir instead of source dir. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--resume', action="store_true", default=False,
                        help='if given, resume from checkpoint(s) of the specified folds.')
    parser.add_argument('--exp_source', type=str, default='experiments/toy_exp',
                        help='specifies, from which source experiment to load configs and data_loader.')
    parser.add_argument('--no_benchmark', action='store_true', help="Do not use cudnn.benchmark.")
    parser.add_argument('-d', '--dev', default=False, action='store_true', help="development mode: shorten everything")

    args = parser.parse_args()
    folds = args.folds

    torch.backends.cudnn.benchmark = not args.no_benchmark

    if args.mode == 'train' or args.mode == 'train_test':

        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, args.use_stored_settings)
        if args.dev:
            folds = [0,1]
            cf.batch_size, cf.num_epochs, cf.min_save_thresh, cf.save_n_models = 3 if cf.dim==2 else 1, 1, 0, 1
            cf.num_train_batches, cf.num_val_batches, cf.max_val_patients = 5, 1, 1
            cf.test_n_epochs =  cf.save_n_models
            cf.max_test_patients = 1

        cf.data_dest = args.data_dest
        logger = utils.get_logger(cf.exp_dir, cf.server_env)
        logger.info("cudnn benchmark: {}, deterministic: {}.".format(torch.backends.cudnn.benchmark,
                                                                     torch.backends.cudnn.deterministic))
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        model = utils.import_module('model', cf.model_path)
        logger.info("loaded model from {}".format(cf.model_path))
        if folds is None:
            folds = range(cf.n_cv_splits)

        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            cf.fold = fold
            cf.resume = args.resume
            if not os.path.exists(cf.fold_dir):
                os.mkdir(cf.fold_dir)
            logger.set_logfile(fold=fold)
            train(logger)
            cf.resume = False
            if args.mode == 'train_test':
                test(logger)

    elif args.mode == 'test':

        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)
        if args.dev:
            folds = [0,1]
            cf.test_n_epochs =  1; cf.max_test_patients = 1

        cf.data_dest = args.data_dest
        logger = utils.get_logger(cf.exp_dir, cf.server_env)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        model = utils.import_module('model', cf.model_path)
        logger.info("loaded model from {}".format(cf.model_path))
        if folds is None:
            folds = range(cf.n_cv_splits)

        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            cf.fold = fold
            logger.set_logfile(fold=fold)
            test(logger)


    # load raw predictions saved by predictor during testing, run aggregation algorithms and evaluation.
    elif args.mode == 'analysis':
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)
        logger = utils.get_logger(cf.exp_dir, cf.server_env)

        if cf.hold_out_test_set:
            cf.folds = args.folds
            predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
            results_list = predictor.load_saved_predictions(apply_wbc=True)
            utils.create_csv_output([(res_dict["boxes"], pid) for res_dict, pid in results_list], cf, logger)

            logger.info('starting evaluation...')
            cf.fold = "overall"
            evaluator = Evaluator(cf, logger, mode='test')
            evaluator.evaluate_predictions(results_list)
            evaluator.score_test_df()
        else:
            fold_dirs = sorted([os.path.join(cf.exp_dir, f) for f in os.listdir(cf.exp_dir) if
                         os.path.isdir(os.path.join(cf.exp_dir, f)) and f.startswith("fold")])
            if folds is None:
                folds = range(cf.n_cv_splits)
            for fold in folds:
                cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
                cf.fold = fold
                logger.set_logfile(fold=fold)
                if cf.fold_dir in fold_dirs:
                    predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
                    results_list = predictor.load_saved_predictions(apply_wbc=True)
                    logger.info('starting evaluation...')
                    evaluator = Evaluator(cf, logger, mode='test')
                    evaluator.evaluate_predictions(results_list)
                    evaluator.score_test_df()
                else:
                    logger.info("Skipping fold {} since no model parameters found.".format(fold))

    # create experiment folder and copy scripts without starting job.
    # useful for cloud deployment where configs might change before job actually runs.
    elif args.mode == 'create_exp':
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, use_stored_settings=False)
        logger = utils.get_logger(cf.exp_dir)
        logger.info('created experiment directory at {}'.format(cf.exp_dir))

    else:
        raise RuntimeError('mode specified in args is not implemented...')

    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    logger.info("{} total runtime: {}".format(os.path.split(__file__)[1], t))
    del logger