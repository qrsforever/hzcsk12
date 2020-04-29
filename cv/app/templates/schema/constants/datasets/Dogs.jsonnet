{
    dataset: 'default',
    task: 'cls',
    method: 'image_classifier',
    data: {
        num_records: 20580,
        num_classes: 120,
        data_dir: '/datasets/Dogs',
        image_tool: 'pil',
        input_mode: 'RGB',
        workers: 1,
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
                96,
                96,
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
                96,
                96,
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
                96,
                96,
            ],
            align_method: 'scale_and_pad',
        },
    },
    network: {
        model_name: 'base_model',
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
                'Chihuahua',
                'Japanese_spaniel',
                'Maltese_dog',
                'Pekinese',
                'Shih-Tzu',
                'Blenheim_spaniel',
                'papillon',
                'toy_terrier',
                'Rhodesian_ridgeback',
                'Afghan_hound',
                'basset',
                'beagle',
                'bloodhound',
                'bluetick',
                'black-and-tan_coonhound',
                'Walker_hound',
                'redbone',
                'borzoi',
                'Irish_wolfhound',
                'Italian_greyhound',
                'whippet',
                'Ibizan_hound',
                'Norwegian_elkhound',
                'Saluki',
                'Scottish_deerhound',
                'Weimaraner',
                'Staffordshire_bullterrier',
                'American_Staffordshire_terrier',
                'Bedlington_terrier',
                'Border_terrier',
                'Kerry_blue_terrier',
                'Irish_terrier',
                'Norfolk_terrier',
                'Norwich_terrier',
                'Yorkshire_terrier',
                'wire-haired_fox_terrier',
                'Lakeland_terrier',
                'Sealyham_terrier',
                'Airedale',
                'cairn',
                'Australian_terrier',
                'Dandie_Dinmont',
                'Boston_bull',
                'miniature_schnauzer',
                'giant_schnauzer',
                'standard_schnauzer',
                'Scotch_terrier',
                'Tibetan_terrier',
                'silky_terrier',
                'soft-coated_wheaten_terrier',
                'West_Highland_white_terrier',
                'Lhasa',
                'flat-coated_retriever',
                'curly-coated_retriever',
                'golden_retriever',
                'Labrador_retriever',
                'Chesapeake_Bay_retriever',
                'German_short-haired_pointer',
                'vizsla',
                'English_setter',
                'Irish_setter',
                'Gordon_setter',
                'Brittany_spaniel',
                'clumber',
                'English_springer',
                'Welsh_springer_spaniel',
                'cocker_spaniel',
                'Sussex_spaniel',
                'Irish_water_spaniel',
                'kuvasz',
                'schipperke',
                'groenendael',
                'malinois',
                'briard',
                'kelpie',
                'komondor',
                'Old_English_sheepdog',
                'collie',
                'Border_collie',
                'Bouvier_des_Flandres',
                'Rottweiler',
                'German_shepherd',
                'Doberman',
                'miniature_pinscher',
                'Greater_Swiss_Mountain_dog',
                'Bernese_mountain_dog',
                'Appenzeller',
                'EntleBucher',
                'boxer',
                'bull_mastiff',
                'Tibetan_mastiff',
                'Great_Dane',
                'Saint_Bernard',
                'Eskimo_dog',
                'malamute',
                'Siberian_husky',
                'affenpinscher',
                'basenji',
                'pug',
                'Leonberg',
                'Newfoundland',
                'Great_Pyrenees',
                'Samoyed',
                'Pomeranian',
                'chow',
                'keeshond',
                'Brabancon_griffon',
                'Pembroke',
                'Cardigan',
                'toy_poodle',
                'miniature_poodle',
                'standard_poodle',
                'Mexican_hairless',
                'dingo',
                'dhole',
                'African_hunting_dog',
                'English_foxhound',
                'otterhound',
                'Shetland_sheepdog',
                'French_bulldog',
            ],
        },
        data: {
            sample: {
                n0: 'imgs/test/n02085620_10621.jpg',
                n1: 'imgs/test/n02085782_1521.jpg',
                n2: 'imgs/test/n02085936_11653.jpg',
                n3: 'imgs/test/n02086079_13647.jpg',
                n4: 'imgs/test/n02086240_11139.jpg',
                n5: 'imgs/test/n02086646_1077.jpg',
                n6: 'imgs/test/n02086910_1659.jpg',
                n7: 'imgs/test/n02087046_1792.jpg',
                n8: 'imgs/test/n02087394_10588.jpg',
                n9: 'imgs/test/n02088094_1128.jpg',
            },
        },
    },
}
