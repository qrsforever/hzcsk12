{
    dataset: 'default',
    task: 'cls',
    method: 'image_classifier',
    data: {
        num_records: 1460,
        num_classes: 9,
        data_dir: '/datasets/Boats',
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
                64,
                64,
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
                64,
                64,
            ],
            align_method: 'scale_and_pad',
        },
    },
    test: {
    },
    network: {
        model_name: 'base_model',
        backbone: 'vgg19',
        distributed: true,
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
            name_seq: [
                'buoy',
                'cruise ship',
                'ferry boat',
                'freight boat',
                'gondola',
                'inflatable boat',
                'kayak',
                'paper boat',
                'sailboat',
            ],
        },
        data: {
            sample: {
                n0: 'imgs/test/buoy#buoy-seagull-water-nature-birds-3764298.jpg',
                n1: 'imgs/test/cruise_ship#aida-ship-driving-cruise-ship-sea-51186.jpg',
                n2: 'imgs/test/ferry_boat#river-ferry-water-ship-boat-1615351.jpg',
                n3: 'imgs/test/freight_boat#cargo-ship-ocean-sea-inland-1101391.jpg',
                n4: 'imgs/test/gondola#venice-italy-gondola-outdoor-1602993.jpg',
                n5: 'imgs/test/inflatable_boat#zachranari-inflatable-boat-ocean-846324.jpg',
                n6: 'imgs/test/kayak#paddle-kayak-canoeing-explore-3729596.jpg',
                n7: 'imgs/test/paper_boat#paper-boat-coloured-colored-2770974.jpg',
                n8: 'imgs/test/sailboat#sail-sailboat-vessel-yacht-charter-863082.jpg',
                n9: 'imgs/test/sailboat#greece-sea-ocean-mountains-2693408.jpg',
            },
        },
    },
}
