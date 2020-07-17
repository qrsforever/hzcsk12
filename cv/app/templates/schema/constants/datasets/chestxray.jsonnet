{
    dataset: 'default',
    task: 'cls',
    method: 'image_classifier',
    data: {
        num_records: 402,
        num_classes: 2,
        data_dir: '/datasets/chestxray',
        image_tool: 'pil',
        input_mode: 'RGB',
        workers: 1,
        normalize: {
            div_value: 255,
            mean: [
                0.4857,
                0.4854,
                0.4882,
            ],
            std: [
                0.2438,
                0.2433,
                0.2442,
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
        display_iter: 4,
        save_iters: 40,
        test_interval: 16,
        max_epoch: 30,
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
            name_seq: ['0', '1'],
        },
        data: {
            sample: {
                n0: 'imgs/normal/IM-0354-0001.jpeg',
                n1: 'imgs/covid/nejmc2001573_f1b.jpeg',
                n2: 'imgs/normal/IM-0419-0001.jpeg',
                n3: 'imgs/covid/covid-19-pneumonia-12.jpg',
                n4: 'imgs/normal/NORMAL2-IM-1234-0001.jpeg',
                n5: 'imgs/covid/kjr-21-e24-g001-l-a.jpg',
                n6: 'imgs/normal/NORMAL2-IM-0572-0001.jpeg',
                n7: 'imgs/covid/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg',
                n8: 'imgs/normal/IM-0499-0001-0002.jpeg',
                n9: 'imgs/covid/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg',
            },
        },
    },
}
