{
    dataset: 'default',
    task: 'cls',
    method: 'image_classifier',
    data: {
        num_records: 70240,
        num_classes: 10,
        data_dir: '/datasets/kannada',
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
                28,
                28,
            ],
            align_method: 'scale_and_pad',
        },
    },
    test: {
        data_transformer: {
            size_mode: 'fix_size',
            input_size: [
                28,
                28,
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
        display_iter: 20,
        save_iters: 200,
        test_interval: 50,
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
            name_seq: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        },
        data: {
            sample: {
                n0: 'imgs/test/00000.jpg',
                n1: 'imgs/test/09311.jpg',
                n2: 'imgs/test/01992.jpg',
                n3: 'imgs/test/08533.jpg',
                n4: 'imgs/test/06514.jpg',
                n5: 'imgs/test/06815.jpg',
                n6: 'imgs/test/08326.jpg',
                n7: 'imgs/test/18817.jpg',
                n8: 'imgs/test/10578.jpg',
                n9: 'imgs/test/22659.jpg',
            },
        },
    },
}
