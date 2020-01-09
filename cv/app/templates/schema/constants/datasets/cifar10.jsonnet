{
    get(dataset_path, checkpt_root):: {
        dataset: 'default',
        task: 'cls',
        method: 'image_classifier',
        data: {
            num_records: 60000,
            num_classes: 10,
            data_dir: dataset_path,
            image_tool: 'pil',
            input_mode: 'RGB',
            workers: 1,
            normalize: {
                div_value: 255,
                mean: [
                    0.485,
                    0.456,
                    0.406,
                ],
                std: [
                    0.229,
                    0.224,
                    0.225,
                ],
            },
        },
        train: {
            batch_size: 32,
            aug_trans: {
                trans_seq: [
                    'random_hflip',
                    'random_border',
                    'random_crop',
                ],
                random_hflip: {
                    ratio: 0.5,
                    swap_pair: [],
                },
                random_border: {
                    ratio: 1,
                    pad: [
                        4,
                        4,
                        4,
                        4,
                    ],
                    allow_outside_center: false,
                },
                random_crop: {
                    ratio: 1,
                    crop_size: [
                        32,
                        32,
                    ],
                    method: 'random',
                    allow_outside_center: false,
                },
            },
            data_transformer: {
                size_mode: 'fix_size',
                input_size: [
                    32,
                    32,
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
                    32,
                    32,
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
                base_lr: 0.1,
                lr_policy: 'multistep',
                step: {
                    gamma: 0.1,
                    step_size: 30,
                },
                multistep: {
                    gamma: 0.1,
                    stepvalue: [
                        150,
                        250,
                        350,
                    ],
                },
            },
            optim: {
                optim_method: 'sgd',
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
            save_iters: 2000,
            test_interval: 100,
            max_epoch: 360,
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
