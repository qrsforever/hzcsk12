#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file train.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-01-20 11:45


import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from k12ai.common.log_message import MessageMetric, MessageReport
from k12ai.common.util_misc import print_options


def main(opt):
    print(opt.dataroot, opt.dataroot.split('/')[-1])
    opt.name = opt.dataroot.split('/')[-1]
    print_options(opt)

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)
    model.setup(opt)
    total_iters = 0
    lr = opt.lr
    mm = MessageMetric()
    num_epochs = opt.n_epochs + opt.n_epochs_decay
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                print(epoch, epoch_iter, losses, lr, '\n')

                mm.add_scalar('训练', '学习率', x=epoch_iter, y=lr).send()
                mm.add_scalar('训练', '损失', x=epoch_iter, y=losses).send()

            if total_iters % opt.save_latest_freq == 0:
                model.save_networks('latest')

        model.save_networks('latest')
        lr = model.update_learning_rate()

        print('End of epoch %d / %d \t lr: %.7f Time Taken: %d sec\n' % (
            epoch, num_epochs, lr,
            time.time() - epoch_start_time))
        mm.add_scalar('训练', '进度', x=epoch, y=round(epoch / num_epochs, 2))


if __name__ == '__main__':
    opt = TrainOptions().parse()
    MessageReport.status(MessageReport.RUNNING)
    try:
        main(opt)
        MessageReport.status(MessageReport.FINISH)
    except Exception:
        MessageReport.status(MessageReport.EXCEPT)
        time.sleep(1)
    finally:
        pass
