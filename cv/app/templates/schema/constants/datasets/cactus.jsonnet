{
    dataset: 'default',
    task: 'cls',
    method: 'image_classifier',
    data: {
        num_records: 17500,
        num_classes: 2,
        data_dir: '/datasets/cactus',
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
                32,
                32,
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
                32,
                32,
            ],
            align_method: 'scale_and_pad',
        },
    },
    test: {
        data_transformer: {
            size_mode: 'fix_size',
            input_size: [
                32,
                32,
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
            name_seq: ['0', '1'],
        },
        data: {
            sample: {
                n0: 'imgs/test/1d0ece89b2213d2daa90d6806f3d07f7.jpg',
                n1: 'imgs/test/f6e47ae1cf5dee5aa6eba0750e60ae9d.jpg',
                n2: 'imgs/test/a15b2ee6da1431a832faa8a68ad1f534.jpg',
                n3: 'imgs/test/91f038f9eb49364dc33d8361998b0a99.jpg',
                n4: 'imgs/test/c2b254219f3395a193c6e62e6ac90480.jpg',
                n5: 'imgs/test/addf4749d3fe82907c39cd441cd59476.jpg',
                n6: 'imgs/test/503f3a8b5118471f247157924e3f31ec.jpg',
                n7: 'imgs/test/ac0a71d4bdc6960af1b4739146d62b93.jpg',
                n8: 'imgs/test/9f9cacf3e65127e007b6223641c0962c.jpg',
                n9: 'imgs/test/9fd67ed5251bf4a898978cc3ef1a0a33.jpg',
            },
        },
    },
}
