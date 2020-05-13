// @file VOC07+12_DET.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-13 21:35

{
    dataset: 'default',
    task: 'det',
    method: 'single_shot_detector',
    data: {
        num_classes: 6,
        num_records: 5543,
        data_dir: '/datasets/underwater',
        image_tool: 'cv2',
        input_mode: 'BGR',
        keep_difficult: false,
        workers: 1,
        normalize: {
            div_value: 1,
            mean: [0.2508, 0.5752, 0.3309],
            std: [0.0523, 0.1089, 0.0726],
        },
    },
    train: {
        batch_size: 16,
        aug_trans: {
            shuffle_trans_seq: ['random_contrast', 'random_hue', 'random_saturation', 'random_brightness', 'random_perm'],
            trans_seq: ['random_hflip', 'random_pad', 'random_det_crop'],
            random_saturation: {
                ratio: 0.5,
                lower: 0.5,
                upper: 1.5,
            },
            random_hue: {
                ratio: 0.5,
                delta: 18,
            },
            random_contrast: {
                ratio: 0.5,
                lower: 0.5,
                upper: 1.5,
            },
            random_pad: {
                ratio: 0.6,
                up_scale_range: [1.0, 4.0],
            },
            random_brightness: {
                ratio: 0.5,
                shift_value: 32,
            },
            random_perm: {
                ratio: 0.5,
            },
            random_hflip: {
                ratio: 0.5,
                swap_pair: [],
            },
            random_det_crop: {
                ratio: 1.0,
            },
        },
        data_transformer: {
            size_mode: 'fix_size',
            input_size: [300, 300],
            align_method: 'scale_and_pad',
        },
    },
    val: {
        batch_size: 16,
        aug_trans: {
            trans_seq: [],
        },
        data_transformer: {
            size_mode: 'fix_size',
            input_size: [300, 300],
            align_method: 'scale_and_pad',
        },
    },
    test: {
        batch_size: 16,
        aug_trans: {
            trans_seq: [],
        },
        data_transformer: {
            size_mode: 'fix_size',
            input_size: [300, 300],
            align_method: 'scale_and_pad',
        },
    },
    res: {
        nms: {
            max_threshold: 0.45,
            pre_nms: 1000,
        },
        val_conf_thre: 0.01,
        vis_conf_thre: 0.5,
        max_per_image: 200,
        cls_keep_num: 50,
    },
    anchor: {
        anchor_method: 'ssd',
        iou_threshold: 0.5,
        num_anchor_list: [4, 6, 6, 6, 4, 4],
        feature_maps_wh: [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]],
        cur_anchor_sizes: [30, 60, 111, 162, 213, 264, 315],
        aspect_ratio_list: [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    },
    network: {
        model_name: 'vgg16_ssd300',
        backbone: 'vgg16',
        num_feature_list: [512, 1024, 512, 256, 256, 256],
        stride_list: [8, 16, 30, 60, 100, 300],
        head_index_list: [0, 1, 2, 3, 4, 5],
        distributed: true,
        gather: true,
        resume_continue: false,
        resume_strict: false,
        resume_val: false,
        custom_model: false,
    },
    solver: {
        lr: {
            metric: 'epoch',
            is_warm: true,
            warm: {
                warm_iters: 1000,
                power: 1.0,
                freeze_backbone: false,
            },
            base_lr: 0.001,
            lr_policy: 'multistep',
            multistep: {
                gamma: 0.1,
                stepvalue: [156, 195, 234],
            },
        },
        optim: {
            optim_method: 'sgd',
            sgd: {
                weight_decay: 0.0005,
                momentum: 0.9,
                nesterov: false,
            },
        },
        display_iter: 20,
        save_iters: 500,
        test_interval: 500,
        max_epoch: 30,
    },
    loss: {
        loss_type: 'multibox_loss',
        loss_weights: {
            multibox_loss: {
                multibox_loss: 1.0,
            },
        },
    },
    _k12: {
        detail: {
            name_seq: [
                'holothurian',
                'echinus',
                'scallop',
                'starfish',
                'waterweeds',
            ],
        },
        data: {
            sample: {
                n0: 'train/image/000692.jpg',
                n1: 'train/image/004682.jpg',
                n2: 'train/image/003302.jpg',
                n3: 'train/image/003102.jpg',
                n4: 'train/image/001010.jpg',
                n5: 'train/image/001669.jpg',
                n6: 'train/image/001494.jpg',
                n7: 'train/image/002442.jpg',
                n8: 'train/image/000232.jpg',
                n9: 'train/image/004658.jpg',
            },
        },
    },
}
