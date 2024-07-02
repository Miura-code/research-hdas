# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

'''
H^c-DAS
search specific cells of different stages.
'''
import os
from config.searchCell_config import SearchCellConfig
from genotypes.genotypes import save_DAG
from trainer.searchCell_trainer import SearchCellTrainer
from utils.logging_util import get_std_logging
from utils.visualize import plot, plot2


def run_task(config):
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.name)))
    config.logger = logger

    config.print_params(logger.info)
    trainer = SearchCellTrainer(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch

    previous_arch = micro_arch = trainer.model.genotype()
    DAG_path = os.path.join(config.DAG_path, "EP00")
    plot_path = os.path.join(config.plot_path, "EP00")
    caption = "Initial DAG"
    plot(micro_arch.normal1, plot_path + '-normal1', caption)
    plot(micro_arch.normal2, plot_path + '-normal2', caption)
    plot(micro_arch.normal3, plot_path + '-normal3', caption)
    plot(micro_arch.reduce1, plot_path + '-reduce1', caption)
    plot(micro_arch.reduce2, plot_path + '-reduce2', caption)
    save_DAG(micro_arch, DAG_path)

    best_top1 = 0.
    for epoch in range(start_epoch, trainer.total_epochs):
        trainer.train_epoch(epoch, printer=logger.info)
        top1 = trainer.val_epoch(epoch, printer=logger.info)
        trainer.lr_scheduler.step()

        # ======== save genotypes ============
        micro_arch = trainer.model.genotype()
        logger.info("genotype = {}".format(micro_arch))
        DAG_path = os.path.join(config.DAG_path, "EP{:02d}".format(epoch + 1))
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
        caption = "Epoch {}".format(epoch + 1)
        plot(micro_arch.normal1, plot_path + "-normal1", caption)
        plot(micro_arch.reduce1, plot_path + "-reduce1", caption)
        plot(micro_arch.normal2, plot_path + "-normal2", caption)
        plot(micro_arch.reduce2, plot_path + "-reduce2", caption)
        plot(micro_arch.normal3, plot_path + "-normal3", caption)

        if best_top1 < top1:
            best_top1, is_best = top1, True
            best_genotype = micro_arch
        else:
            is_best = False
        trainer.save_checkpoint(epoch, is_best=is_best)
        if previous_arch != micro_arch:
            save_DAG(micro_arch, DAG_path, is_best=is_best)
        previous_arch = micro_arch
        
        logger.info("Until now, best Prec@1 = {:.4%}".format(best_top1))
    
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Final Best Genotype = {}".format(best_genotype))


def main():
    config = SearchCellConfig()
    run_task(config)


if __name__ == "__main__":
    main()
