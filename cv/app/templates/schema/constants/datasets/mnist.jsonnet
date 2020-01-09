{
    get(dataset_path, checkpt_root):: {
        dataset: 'default',
        task: 'cls',
        method: 'image_classifier',
        data: {
            num_records: 70000,
            num_classes: 10,
            data_dir: dataset_path,
            image_tool: 'cv2',
            input_mode: 'BGR',
            workers: 1,
            normalize: {
                div_value: 1,
                mean: [
                    0.1307,
                    0.1307,
                    0.1307,
                ],
                std: [
                    0.3081,
                    0.3081,
                    0.3081,
                ],
            },
        },
        train: {
            batch_size: 32,
            aug_trans: {
                trans_seq: [],
            },
            data_transformer: {
                size_mode: 'fix_size',
                input_size: [
                    28,
                    28,
                ],
                align_method: 'only_pad',
            },
        },
        val: {
            batch_size: 32,
            aug_trans: {
                trans_seq: [],
            },
            data_transformer: {
                size_mode: 'fix_size',
                input_size: [
                    28,
                    28,
                ],
                align_method: 'only_pad',
            },
        },
        test: {
        },
        network: {
            model_name: 'base_model',
            backbone: 'vgg19',
            distributed: true,
            gather: true,
            checkpoints_dir: 'ckpts',
            checkpoints_root: checkpt_root,
        },
        solver: {
            lr: {
                metric: 'epoch',
                base_lr: 0.0001,
                lr_policy: 'step',
                step: {
                    gamma: 0.1,
                    step_size: 30,
                },
                multistep: {},
            },
            optim: {
                optim_method: 'adam',
                adam: {
                    betas: [
                        0.9,
                        0.999,
                    ],
                    eps: 1e-8,
                    weight_decay: 0.0001,
                },
                sgd: {
                    weight_decay: 0.00004,
                    momentum: 0.9,
                    nesterov: false,
                },
            },
            display_iter: 20,
            save_iters: 200,
            test_interval: 50,
            max_epoch: 60,
        },
        loss: {
            loss_type: 'ce_loss',
            loss_weights: {
                ce_loss: {
                    ce_loss: 1,
                },
                soft_ce_loss: {
                    soft_ce_loss: 1,
                },
                mixup_ce_loss: {
                    mixup_ce_loss: 1,
                },
                mixup_soft_ce_loss: {
                    mixup_soft_ce_loss: 1,
                },
            },
            params: {
                ce_loss: {
                    reduction: 'mean',
                    ignore_index: -1,
                },
                soft_ce_loss: {
                    reduction: 'batchmean',
                    label_smooth: 0.1,
                },
                mixup_ce_loss: {
                    reduction: 'mean',
                    ignore_index: -1,
                },
                mixup_soft_ce_loss: {
                    reduction: 'batchmean',
                    label_smooth: 0.1,
                },
            },
        },
    },
}
