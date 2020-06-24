{
    task: 'make3d',
    data: {
        dataset_root: '/datasets/nyu/',
        dataset_name: 'nyu',
        dataset_loader: {
            type: 'listdir',
            args: {
                batch_size: 32,
                num_workers: 2,
                shuffle: true,
                drop_last: true,
                pin_memory: true,
            },
        },
        transforms: {
            output_size: [
                228,
                304,
            ],
            normalize: {
                mean: [
                    0.5,
                    0.5,
                    0.5,
                ],
                std: [
                    0.5,
                    0.5,
                    0.5,
                ],
            },
            shuffle: false,
            compose: {
                resize: {
                    args: {
                        size: [
                            250,
                            250,
                        ],
                        interpolation: 2,
                    },
                    phase: {
                        train: true,
                        valid: false,
                        evaluate: false,
                    },
                },
                center_crop: {
                    args: {
                        size: [
                            238,
                            238,
                        ],
                    },
                    phase: {
                        train: true,
                        valid: false,
                        evaluate: false,
                    },
                },
                random_horizontal_flip: {
                    args: {
                        p: 0.5,
                    },
                    phase: {
                        train: true,
                        valid: false,
                        evaluate: false,
                    },
                },
                color_jitter: {
                    args: {
                        brightness: 0.5,
                        contrast: 0.5,
                        saturation: 0.5,
                        hue: 0.5,
                    },
                    phase: {
                        train: true,
                        valid: false,
                        evaluate: false,
                    },
                },
            },
        },
    },
    _k12: {
        data: {
            transforms: {
                compose: {
                    resize: true,
                    center_crop: true,
                    random_horizontal_flip: true,
                    color_jitter: true,
                },
            },
        },
    },
    model: {
        network: 'fcrn',
        distributed: false,
        resume: false,
        backbone: 'resnet50',
        pretrained: false,
    },
    hypes: {
        epoch: 100,
        criterion: {
            type: 'maskedmse',
        },
        optimizer: {
            type: 'adam',
            args: {
                lr: 0.001,
                weight_decay: 0.001,
                betas: [
                    0.5,
                    0.999,
                ],
                eps: 1e-08,
            },
        },
        scheduler: {
            type: 'step',
            args: {
                gamma: 0.1,
                step_size: 2,
            },
        },
    },
}
