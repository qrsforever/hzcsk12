{
    dataset: 'default',
    task: 'cls',
    method: 'image_classifier',
    data: {
        num_records: 25000,
        num_classes: 2,
        data_dir: '/datasets/dogsVsCats',
        image_tool: 'cv2',
        input_mode: 'BGR',
        workers: 4,
        normalize: {
            div_value: 255,
            mean: [
                0.485,
                0.456,
                0.406,
            ],
            std: [
                0.485,
                0.456,
                0.406,
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
                128,
                128,
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
                128,
                128,
            ],
            align_method: 'scale_and_pad',
        },
    },
    test: {
        batch_size: 32,
        aug_trans: {
            trans_seq: [],
        },
        data_transformer: {
            size_mode: 'fix_size',
            input_size: [
                128,
                128,
            ],
            align_method: 'scale_and_pad',
        },
    },
    network: {
        model_name: 'base_model',
        gather: true,
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
            name_seq: ['dog', 'cat'],
        },
        data: {
            sample: {
                n0: 'imgs/test/cat.11286.jpg',
                n1: 'imgs/test/dog.216.jpg',
                n2: 'imgs/test/cat.12256.jpg',
                n3: 'imgs/test/dog.2186.jpg',
                n4: 'imgs/test/cat.11646.jpg',
                n5: 'imgs/test/dog.2566.jpg',
                n6: 'imgs/test/cat.1286.jpg',
                n7: 'imgs/test/dog.3196.jpg',
                n8: 'imgs/test/dog.3386.jpg',
                n9: 'imgs/test/cat.11486.jpg',
            },
        },
    },
}
