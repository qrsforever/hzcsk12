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
                    if method == 'random_brightness'
                    then (import 'random/random_brightness.libsonnet').get(jid + '.aug_trans.random_brightness')
                    else if method == 'random_contrast'
                    then (import 'random/random_contrast.libsonnet').get(jid + '.aug_trans.random_contrast')
                    else if method == 'random_hue'
                    then (import 'random/random_hue.libsonnet').get(jid + '.aug_trans.random_hue')
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
        _aug_trans_group_item('train', 'random_brightness', 'Random Brightness'),
        _aug_trans_group_item('train', 'random_contrast', 'Random Contrast'),
        _aug_trans_group_item('train', 'random_hue', 'Random Hue'),
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
