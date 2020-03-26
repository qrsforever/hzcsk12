{
    dataset: 'default',
    task: 'cls',
    method: 'image_classifier',
    data: {
        num_records: 438204,
        num_classes: 1138,
        data_dir: '/datasets/cellular',
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
        batch_size: 16,
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
    val: {
        batch_size: 16,
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
        batch_size: 16,
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
            name_seq: [],
        },
        data: {
            sample: {
                n0: 'imgs/train/HEPG2-01/Plate1/B04_s2_w2.png',
                n1: 'imgs/train/HEPG2-01/Plate1/B07_s2_w1.png',
                n2: 'imgs/train/HEPG2-03/Plate3/C12_s1_w4.png',
                n3: 'imgs/train/HUVEC-02/Plate4/I03_s1_w2.png',
                n4: 'imgs/train/HUVEC-08/Plate4/G20_s1_w2.png',
                n5: 'imgs/train/HUVEC-09/Plate1/B20_s2_w3.png',
                n6: 'imgs/train/HUVEC-13/Plate1/H02_s1_w1.png',
                n7: 'imgs/train/RPE-04/Plate3/E20_s2_w2.png',
                n8: 'imgs/train/U2OS-03/Plate2/D19_s1_w5.png',
                n9: 'imgs/train/HEPG2-05/Plate1/N21_s2_w5.png',
            },
        },
    },
}
