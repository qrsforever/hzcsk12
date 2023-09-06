{
    dataset: 'default',
    task: 'cls',
    method: 'image_classifier',
    data: {
        num_records: 29250,
        num_classes: 35,
        data_dir: '/datasets/rcnfood35',
        image_tool: 'pil',
        input_mode: 'RGB',
        workers: 1,
        normalize: {
            div_value: 1,
            mean: [0.6332, 0.5754, 0.4965],
            std: [0.2632, 0.2896, 0.3205],
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
			name_seq: [
				'麻婆豆腐',
				'酸辣土豆丝',
				'鱼香茄子',
                '醋溜白菜',
				'手撕包菜',
				'炒豆芽',
				'蚝油西兰花',
				'莲藕',
				'凉拌木耳',
				'花生米',
				'炒苦瓜',
				'松仁玉米',
				'西红柿炒鸡蛋',
				'猪肝',
				'糖醋排骨',
				'可乐鸡翅',
				'啤酒鸭',
				'红烧肉',
				'梅菜扣肉',
				'酱焖猪蹄',
				'蚂蚁上树',
				'羊肉串',
				'酸菜鱼',
				'香辣小龙虾',
				'生蚝',
				'鸡蛋灌饼',
				'馒头',
				'炸酱面',
				'饺子',
				'小米粥',
				'皮蛋瘦肉粥',
				'毛血旺',
				'北京烤鸭',
				'麻花',
				'凉拌皮蛋'
            ]
        },
        data: {
            sample: {
            },
        },
    },
}
