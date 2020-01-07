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
};

local _data_transform(jid, label) = {
    name: { en: label + ' Data Transform', cn: self.en },
    type: 'object',
    objs: [
        _Utils.int(jid + '.fit_stride', 'Fit Stride', def=1),
        {
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
                },
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
                },
            ],
        },
    ],
};

local _aug_trans_group_item(jid, method, display) = {
    _id_: '_k12.' + jid + '.' + method + '.bool',
    name: { en: display, cn: self.en },
    type: 'bool-trigger',
    objs: [
        {
            value: true,
            trigger: {
                objs: [
                    {
                        _id_: '_k12._stringarray_.' + jid + '.' + method,
                        name: { en: 'Transform Type', cn: self.en },
                        type: 'string-enum',
                        objs: [
                            {
                                name: { en: 'Normal', cn: self.en },
                                value: jid + '.aug_trans.trans_seq',
                            },
                            {
                                name: { en: 'Shuffle', cn: self.en },
                                value: jid + '.aug_trans.shuffle_trans_seq',
                            },
                        ],
                        default: self.objs[0].value,
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
    name: { en: label + ' Augment Transform', cn: self.en },
    type: 'H',
    objs: [
        _aug_trans_group_item('train', 'random_border', 'Random Border'),
        _aug_trans_group_item('train', 'random_brightness', 'Random Brightness'),
        _aug_trans_group_item('train', 'random_contrast', 'Random Contrast'),
        _aug_trans_group_item('train', 'random_crop', 'Random Crop'),
        _aug_trans_group_item('train', 'random_det_crop', 'Random Det Crop'),
        _aug_trans_group_item('train', 'random_gauss_blur', 'Random Gauss Blur'),
        _aug_trans_group_item('train', 'random_hsv', 'Random HSV'),
        _aug_trans_group_item('train', 'random_hue', 'Random Hue'),
        _aug_trans_group_item('train', 'random_pad', 'Random Pad'),
        _aug_trans_group_item('train', 'random_perm', 'Random Perm'),
        _aug_trans_group_item('train', 'random_resize', 'Random Resize'),
        _aug_trans_group_item('train', 'random_resized_crop', 'Random Resized Crop'),
        _aug_trans_group_item('train', 'random_rotate', 'Random Rotate'),
        _aug_trans_group_item('train', 'random_saturation', 'Random Saturation'),
    ],
};

{
    get():: {
        type: '_ignore_',
        objs: [
            {
                name: { en: 'Phase', cn: self.en },
                type: 'navigation',
                objs: [
                    {
                        name: { en: 'Train', cn: self.en },
                        type: '_ignore_',
                        objs: [
                            _data_transform('train', 'Train'),
                            _augment_transform('train', 'Train'),
                        ],
                    },
                    {
                        name: { en: 'Validation', cn: self.en },
                        type: '_ignore_',
                        objs: [
                            _data_transform('val', 'Validation'),
                            _augment_transform('val', 'Validation'),
                        ],
                    },
                    {
                        name: { en: 'Test', cn: self.en },
                        type: '_ignore_',
                        objs: [
                            _data_transform('test', 'Test'),
                            _augment_transform('test', 'Test'),
                        ],
                    },
                ],
            },
        ],
    },
}
