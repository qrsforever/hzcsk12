{
    dataset: 'default',
    task: 'cls',
    method: 'image_classifier',
    data: {
        num_records: 387,
        num_classes: 6,
        data_dir: '/datasets/roffice6',
        image_tool: 'pil',
        input_mode: 'RGB',
        workers: 1,
        normalize: {
            div_value: 1,
            mean: [0.5851, 0.5585, 0.541],
            std: [0.2335, 0.2391, 0.2447]
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
                224,
                224,
            ],
            align_method: 'scale_and_pad',
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
                224,
                224,
            ],
            align_method: 'scale_and_pad',
        },
    },
    test: {
        data_transformer: {
            size_mode: 'fix_size',
            input_size: [
                224,
                224,
            ],
            align_method: 'scale_and_pad',
        },
    },
    network: {
        model_name: 'base_model',
        backbone: 'resnet18',
        distributed: true,
        gather: true,
    },
    solver: {
        lr: {
            metric: 'epoch',
            base_lr: 0.01,
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
        display_iter: 32,
        save_iters: 128,
        test_interval: 256,
        max_epoch: 50,
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
    _k12: {
        detail: {
            name_seq: ['Eraser', 'Keyboard', 'Knives', 'Laptop', 'Mouse', 'Pencil']
        },
        data: {
            sample: {
            },
        },
    },
}
