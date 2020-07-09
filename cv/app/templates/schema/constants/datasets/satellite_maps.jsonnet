// @file satellite_maps.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-07-09 15:37

{
    dataset: 'default_pix2pix',
    task: 'gan',
    method: 'image_translator',
    data: {
        data_dir: '/datasets/satellite_maps',
        image_tool: 'pil',
        input_mode: 'RGB',
        workers: 2,
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
    },
    train: {
        batch_size: 32,
        aug_trans: {
            trans_seq: ['random_hflip', 'random_crop'],
            random_hflip: {
                ratio: 0.5,
                swap_pair: [],
            },
            random_crop: {
                ratio: 1.0,
                crop_size: [128, 128],
                method: 'random',
                allow_outside_center: false,
            },
        },
        data_transformer: {
            // size_mode: 'none',
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
            trans_seq: ['random_crop'],
            random_crop: {
                ratio: 1.0,
                crop_size: [128, 128],
                method: 'center',
                allow_outside_center: false,
            },
        },
        data_transformer: {
            size_mode: 'none',
        },
    },
    test: {
        batch_size: 32,
        aug_trans: {
            trans_seq: ['random_crop'],
            random_crop: {
                ratio: 1.0,
                crop_size: [128, 128],
                method: 'center',
                allow_outside_center: false,
            },
        },
        data_transformer: {
            size_mode: 'none',
        },
    },
    network: {
        model_name: 'pix2pix',
        backbone: 'none',
        distributed: false,
        use_dropout: false,
        gather: true,
        norm_type: 'batchnorm',
        imgpool_size: 0,
        generator: {
            net_type: 'unet_128',
            init_type: 'normal',
            init_gain: 0.02,
            in_c: 3,
            out_c: 3,
            num_f: 64,
        },
        discriminator: {
            net_type: 'n_layers',
            init_type: 'normal',
            init_gain: 0.02,
            n_layers: 3,
            num_f: 64,
            in_c: 6,
        },
    },
    solver: {
        lr: {
            metric: 'epoch',
            base_lr: 0.00006,
            lr_policy: 'lambda_fixlinear',
            lambda_fixlinear: {
                fix_value: 100,
                linear_value: 100,
            },
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
        loss_type: 'gan_loss',
        loss_weights: {
            l1_loss: 100.0,
            gan_loss: 1.0,
        },
        params: {
            gan_loss: {
                gan_mode: 'vanilla',
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
                n1: 'imgs/test/00001.jpg',
                n2: 'imgs/test/00002.jpg',
                n3: 'imgs/test/00003.jpg',
                n4: 'imgs/test/00004.jpg',
                n5: 'imgs/test/00005.jpg',
                n6: 'imgs/test/00006.jpg',
                n7: 'imgs/test/00007.jpg',
                n8: 'imgs/test/00008.jpg',
                n9: 'imgs/test/00009.jpg',
            },
        },
    },
}
