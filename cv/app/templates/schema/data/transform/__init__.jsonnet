// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 14:46

local _Utils = import '../../utils/helper.libsonnet';

local _pad_mode(jid) = {
    _id_: jid + '.pad_mode',
    name: { en: 'Pad Mode', cn: self.en },
    type: 'string-enum',
    objs: [
        {
            name: { en: 'random', cn: self.en },
            value: 'random',
        },
        {
            name: { en: 'border', cn: self.en },
            value: 'pad_border',
        },
        {
            name: { en: 'left top', cn: self.en },
            value: 'pad_left_up',
        },
        {
            name: { en: 'right bottom', cn: self.en },
            value: 'pad_right_down',
        },
        {
            name: { en: 'center', cn: self.en },
            value: 'pad_center',
        },
    ],
    default: _Utils.get_default_value(self._id_, self.objs[1].value),
};

local _data_transform(jid, label) = {
    name: { en: label + ' Data Transform', cn: self.en },
    type: 'object',
    objs: [  // 1
        _Utils.int(jid + '.fit_stride', 'Fit Stride', def=1),
        {  // 2
            type: 'H',
            objs: [
                {
                    _id_: jid + '.size_mode',
                    name: { en: 'Size Mode', cn: self.en },
                    type: 'string-enum-trigger',
                    objs: [
                        {
                            name: { en: 'none', cn: self.en },
                            value: 'none',
                            trigger: {},
                        },
                        {
                            name: { en: 'fix size', cn: self.en },
                            value: 'fix_size',
                            trigger: {
                                type: '_ignore_',
                                objs: [
                                    _Utils.intarray(jid + '.input_size', 'Input Size', def=[32, 32], ddd=true),
                                ],
                            },
                        },
                        {
                            name: { en: 'multi size', cn: self.en },
                            value: 'multi_size',
                            trigger: {
                                type: '_ignore_',
                                objs: [
                                    _Utils.intarray(jid + '.ms_input_size', 'Multi Input Size', def=[[416, 416]], ddd=true),
                                ],
                            },
                        },
                        {
                            name: { en: 'max size', cn: self.en },
                            value: 'max_size',
                            trigger: {},
                        },
                    ],
                    default: _Utils.get_default_value(self._id_, self.objs[1].value),
                    readonly: true,
                },
            ] + (if _Utils.task == 'gan'
                 then []
                 else [
                     {
                         _id_: jid + '.align_method',
                         name: { en: 'Align Method', cn: self.en },
                         type: 'string-enum-trigger',
                         objs: [
                             {
                                 name: { en: 'only scale', cn: self.en },
                                 value: 'only_scale',
                                 trigger: {},
                             },
                             {
                                 name: { en: 'scale and pad', cn: self.en },
                                 value: 'scale_and_pad',
                                 trigger: {
                                     type: '_ignore_',
                                     objs: [_pad_mode(jid)],
                                 },
                             },
                             {
                                 name: { en: 'only pad', cn: self.en },
                                 value: 'only_pad',
                                 trigger: {
                                     type: '_ignore_',
                                     objs: [_pad_mode(jid)],
                                 },
                             },
                         ],
                         default: _Utils.get_default_value(self._id_, self.objs[2].value),
                         readonly: true,
                     },
                 ]),
        },  // 2
    ],  // 1
};

local _aug_trans_group_item(jid, method, display) = {
    _id_: '_k12.' + jid + '.' + method + '.bool',
    name: { en: display, cn: self.en },
    type: 'bool-trigger',
    objs: [
        {
            value: true,
            trigger: {
                type: 'H',
                objs: [
                    {
                        _id_: '_k12.trans_seq_group.' + jid + '.' + method,
                        name: { en: 'type', cn: self.en },
                        type: 'string-enum',
                        objs: [
                            {
                                name: { en: 'Normal', cn: self.en },
                                value: 'trans_seq',
                            },
                            {
                                name: { en: 'Shuffle', cn: self.en },
                                value: 'shuffle_trans_seq',
                            },
                        ],
                        default: self.objs[0].value,
                        // tips: 'Normal: all random transforms executing in order, Shuffle: all random transforms shutffling before executing',
                    },
                ] + (
                    if method == 'random_border'
                    then (import 'random/random_border.libsonnet').get(jid + '.aug_trans.random_border')
                    else if method == 'random_brightness'
                    then (import 'random/random_brightness.libsonnet').get(jid + '.aug_trans.random_brightness')
                    else if method == 'random_contrast'
                    then (import 'random/random_contrast.libsonnet').get(jid + '.aug_trans.random_contrast')
                    else if method == 'random_crop'
                    then (import 'random/random_crop.libsonnet').get(jid + '.aug_trans.random_crop')
                    else if method == 'random_det_crop'
                    then (import 'random/random_det_crop.libsonnet').get(jid + '.aug_trans.random_det_crop')
                    else if method == 'random_focus_crop'
                    then (import 'random/random_focus_crop.libsonnet').get(jid + '.aug_trans.random_focus_crop')
                    else if method == 'random_hflip'
                    then (import 'random/random_hflip.libsonnet').get(jid + '.aug_trans.random_hflip')
                    else if method == 'random_hsv'
                    then (import 'random/random_hsv.libsonnet').get(jid + '.aug_trans.random_hsv')
                    else if method == 'random_hue'
                    then (import 'random/random_hue.libsonnet').get(jid + '.aug_trans.random_hue')
                    else if method == 'random_pad'
                    then (import 'random/random_pad.libsonnet').get(jid + '.aug_trans.random_pad')
                    else if method == 'random_perm'
                    then (import 'random/random_perm.libsonnet').get(jid + '.aug_trans.random_perm')
                    else if method == 'random_resize'
                    then (import 'random/random_resize.libsonnet').get(jid + '.aug_trans.random_resize')
                    else if method == 'random_resized_crop'
                    then (import 'random/random_resized_crop.libsonnet').get(jid + '.aug_trans.random_resized_crop')
                    else if method == 'random_rotate'
                    then (import 'random/random_rotate.libsonnet').get(jid + '.aug_trans.random_rotate')
                    else if method == 'random_saturation'
                    then (import 'random/random_saturation.libsonnet').get(jid + '.aug_trans.random_saturation')
                    else []
                ),
            },
        },
        {
            value: false,
            trigger: {},
        },
    ],
    default: false,
};

local _augment_transform(jid, label) = {
    name: { en: label + ' Random Transform', cn: self.en },
    type: 'V',
    objs: [
        _aug_trans_group_item(jid, 'random_border', 'Border'),
        _aug_trans_group_item(jid, 'random_brightness', 'Brightness'),
        _aug_trans_group_item(jid, 'random_contrast', 'Contrast'),
        _aug_trans_group_item(jid, 'random_crop', 'Crop'),
        _aug_trans_group_item(jid, 'random_det_crop', 'Det Crop'),
        _aug_trans_group_item(jid, 'random_gauss_blur', 'Gauss Blur'),
        _aug_trans_group_item(jid, 'random_hsv', 'HSV'),
        _aug_trans_group_item(jid, 'random_hue', 'Hue'),
        _aug_trans_group_item(jid, 'random_pad', 'Pad'),
        _aug_trans_group_item(jid, 'random_perm', 'Perm'),
        _aug_trans_group_item(jid, 'random_resize', 'Resize'),
        _aug_trans_group_item(jid, 'random_resized_crop', 'Resized Crop'),
        _aug_trans_group_item(jid, 'random_rotate', 'Rotate'),
        _aug_trans_group_item(jid, 'random_saturation', 'Saturation'),
    ],
};

{
    get():: [
        {
            name: { en: 'Phase', cn: self.en },
            type: 'navigation',
            objs: [
                {
                    name: { en: 'Train', cn: self.en },
                    type: '_ignore_',
                    objs: [
                        _data_transform('train.data_transformer', 'Train'),
                        _augment_transform('train', 'Train'),
                    ],
                },
                {
                    name: { en: 'Validation', cn: self.en },
                    type: '_ignore_',
                    objs: [
                        _data_transform('val.data_transformer', 'Validation'),
                        _augment_transform('val', 'Validation'),
                    ],
                },
                {
                    name: { en: 'Evaluate', cn: self.en },
                    type: '_ignore_',
                    objs: [
                        _data_transform('test.data_transformer', 'Test'),
                        _augment_transform('test', 'Test'),
                    ],
                },
            ],
        },
    ],
}
