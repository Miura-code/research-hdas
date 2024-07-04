# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" Training augmented macro-architecture(stage) model """
import os
import torch
import torch.nn as nn
import numpy as np
from models.augment_cellcnn import AugmentCellCNN
import utils
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.data_util import get_data
from utils.logging_util import get_std_logging
from utils.eval_util import AverageMeter, accuracy
from utils.measurement_utils import TimeKeeper
from utils.file_management import load_evaluated_checkpoint_state
from config.test_config import TestConfig
import utils.measurement_utils


config = TestConfig()

device = torch.device("cuda")

logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.save)))
config.logger = logger
config.print_params(logger.info)

writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

def main():
    logger.info("Logger is set - test start")

    # ================= set gpu ==================
    utils.set_seed_gpu(config.seed, config.gpus[0])

    # ================= get dataset ==================
    input_size, input_channels, n_classes, _, valid_data = get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True)
    # ================= define dataloader ==================
    n_val = len(valid_data)
    split = int(np.floor(config.train_portion * n_val))
    indices = list(range(n_val))
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    test_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               sampler=test_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)

    # ================= define model ==================
    use_aux = config.aux_weight > 0.
    model = AugmentCellCNN(input_size, input_channels, config.init_channels, n_classes, config.layers, use_aux, config.genotype).to(device)
    # ================= load checkpoint ==================
    load_evaluated_checkpoint_state(model=model, optimizer=None, checkpoint_path=config.resume_path)

    logger.info(f"--> Loaded checkpoint '{config.resume_path}'")
    logger.info("param size = %fMB", utils.measurement_utils.count_parameters_in_MB(model))
    mac, params = utils.measurement_utils.count_ModelSize_byptflops(model, (3,32,32))
    logger.info("param size = {}MB, mac = {}".format(params, mac))

    criterion = nn.CrossEntropyLoss().to(device)
    # ================= Test model ==================
    time_keeper = TimeKeeper()
    test_top1, test_top5 = validate(test_loader, model, criterion)
    time_keeper.end()
    start, end, diff = time_keeper.print_info()

    logger.info("Test Prec(@1, @5) = ({:.4%}, {:.4%})".format(test_top1, test_top5))
    logger.info("Time to Test = ({}, {}, {})".format(start, end, diff))

    writer.add_text('result', utils.ListToMarkdownTable(["ACC_TOP1", "ACC_TOP5"], [test_top1, test_top5]), 0)
    writer.close()


def validate(valid_loader, model, criterion):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in tqdm(enumerate(valid_loader)):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = accuracy(logits, y, topk=(1,5))
            # ================= record process ==================
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Test: Step {:03d}/{:03d} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        step, len(valid_loader) - 1, top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == "__main__":
    cudnn.benchmark = True
    main()
