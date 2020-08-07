// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 23:05

local _Utils = import '../utils/helper.libsonnet';

local _gan_pix2pix_netparams(net) = [
    {
        type: 'H',
        objs: [
            {
                _id_: 'network.' + net + '.net_type',
                name: { en: 'Net Type', cn: self.en },
                type: 'string-enum',
                objs: if net == 'generator' then
                    [
                        {
                            name: { en: 'unet_128', cn: self.en },
                            value: 'unet_128',
                        },
                        {
                            name: { en: 'unet_256', cn: self.en },
                            value: 'unet_256',
                        },
                        {
                            name: { en: 'resnet_9', cn: self.en },
                            value: 'resnet_9blocks',
                        },
                        {
                            name: { en: 'resnet_6', cn: self.en },
                            value: 'resnet_6blocks',
                        },
                    ] else [
                    {
                        name: { en: 'n_layers', cn: self.en },
                        value: 'n_layers',
                    },
                    {
                        name: { en: 'fc', cn: self.en },
                        value: 'fc',
                    },
                    {
                        name: { en: 'pixel', cn: self.en },
                        value: 'pixel',
                    },
                ],
                default: _Utils.get_default_value(self._id_, self.objs[0].value),
            },
            {
                _id_: 'network.' + net + '.init_type',
                name: { en: 'Init Type', cn: self.en },
                type: 'string-enum',
                objs: [
                    {
                        name: { en: 'normal', cn: self.en },
                        value: 'normal',
                    },
                    {
                        name: { en: 'xavier', cn: self.en },
                        value: 'xavier',
                    },
                    {
                        name: { en: 'kaiming', cn: self.en },
                        value: 'kaiming',
                    },
                    {
                        name: { en: 'orthogonal', cn: self.en },
                        value: 'orthogonal',
                    },
                ],
                default: _Utils.get_default_value(self._id_, self.objs[0].value),
            },
            _Utils.float('network.' + net + '.init_gain', 'Init Gain', def=0.02, min=0.01),
        ],
    },
    {
        type: 'H',
        objs: [
            _Utils.int('network.' + net + '.num_f', 'Num Filters', def=64, min=8),
            _Utils.int('network.' + net + '.in_c', 'In Channels', def=3, min=1),
        ] + if net == 'generator'
        then [
            _Utils.int('network.' + net + '.out_c', 'Out Channels', def=3, min=1),
        ] else [
            _Utils.int('network.' + net + '.n_layers', 'Num Layers', def=3, min=1),
        ],
    },
];

(import 'networks/__init__.jsonnet').get()
+
[
    {
        type: 'accordion',
        objs:
            if _Utils.method == 'single_shot_detector' then [
                {
                    name: { en: 'SSD', cn: self.en },
                    type: '_ignore_',
                    objs: [
                        {
                            type: 'H',
                            objs: [
                                {
                                    _id_: 'anchor.anchor_method',
                                    name: { en: 'Anchor Method', cn: self.en },
                                    type: 'string-enum',
                                    objs: [
                                        {
                                            name: { en: 'ssd', cn: self.en },
                                            value: 'ssd',
                                        },
                                        {
                                            name: { en: 'retina', cn: self.en },
                                            value: 'retina',
                                        },
                                        {
                                            name: { en: 'naive', cn: self.en },
                                            value: 'naive',
                                        },
                                    ],
                                    default: self.objs[0].value,
                                    readonly: true,
                                },
                                _Utils.float('anchor.iou_threshold', 'IOU Threshold', def=0.5, ddd=true),
                            ],
                        },
                        {
                            type: 'H',
                            objs: [
                                _Utils.float('res.nms.max_threshold', 'NMS Threshold', def=0.5, ddd=true),
                                _Utils.float('res.val_conf_thre', 'Score Threshold', def=0.05, ddd=true),
                                _Utils.float('res.vis_conf_thre', 'Visual Threshold', def=0.5, ddd=true),
                            ],
                        },
                        {
                            type: 'H',
                            objs: [
                                _Utils.int('res.nms.pre_nms', 'NMS Pre', def=1000, ddd=true),
                                _Utils.int('res.max_per_image', 'Max Per Image', def=200, ddd=true),
                                _Utils.int('res.cls_keep_num', 'Max Per Class', def=20, ddd=true),
                            ],
                        },
                    ],
                },
            ] else if _Utils.network == 'pix2pix'
            then [
                {
                    name: { en: 'Generator', cn: self.en },
                    type: '_ignore_',
                    objs: _gan_pix2pix_netparams('generator'),
                },
                {
                    name: { en: 'Discriminator', cn: self.en },
                    type: '_ignore_',
                    objs: _gan_pix2pix_netparams('discriminator'),
                },
            ] else [
                {
                    name: { en: 'Details', cn: self.en },
                    type: '_ignore_',
                    objs: (import 'details/__init__.jsonnet').get(),
                },
            ],
    },
]
