// @file augtrans.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-10 22:42

local _ratio_ = {
    type: 'float',
    name: 'Ratio',
    default: 0.5,
};

{
    aug_trans: {
        type: 'object',
        name: 'Aug Transform',
        description: |||
            todo
        |||,

        shuffle_trans_seq: {
            type: 'enum-array',
            name: 'Shuffle Transform Sequence',

            items: {
                type: 'string',
                values: [
                    {
                        name: 'Random Contrast',
                        value: 'random_contrast',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_contrast',
                    },
                    {
                        name: 'Random Hue',
                        value: 'random_hue',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_hue',
                    },
                    {
                        name: 'Random Satuation',
                        value: 'random_saturation',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_saturation',
                    },
                    {
                        name: 'Random Brightness',
                        value: 'random_brightness',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_brightness',
                    },
                    {
                        name: 'Random Perm',
                        value: 'random_perm',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_perm',
                    },
                ],
            },
            default: [],
        },
        trans_seq: {
            type: 'enum-array',
            name: 'Transform Sequence',
            items: {
                type: 'string',
                values: [
                    {
                        name: 'Random Resize',
                        value: 'random_resize',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_resize',
                    },
                    {
                        name: 'Random Focus Crop',
                        value: 'random_focus_crop',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_focus_crop',
                    },
                    {
                        name: 'Random Rotate',
                        value: 'random_rotate',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_rotate',
                    },
                    {
                        name: 'Random HFlip',
                        value: 'random_hflip',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_hflip',
                    },
                    {
                        name: 'Random Pad',
                        value: 'random_pad',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_pad',
                    },
                    {
                        name: 'Random Det Crop',
                        value: 'random_det_crop',
                        ref: std.extVar('aug_parent') + 'aug_trans.random_det_crop',
                    },
                ],
            },
            default: [],
        },

        random_contrast: {
            type: 'object',
            name: 'Random Contrast Parameters',
            ratio: _ratio_,
            lower: {
                type: 'float',
                default: 0.5,
            },
            upper: {
                type: 'float',
                default: 1.5,
            },
        },

        random_saturation: {
            type: 'object',
            name: 'Random Satuation Parameters',
            ratio: _ratio_,
            lower: {
                type: 'float',
                default: 0.5,
            },
            upper: {
                type: 'float',
                default: 1.5,
            },
        },

        random_hue: {
            type: 'object',
            name: 'Random Hue Parameters',
            ratio: _ratio_,
            delta: {
                type: 'int',
                default: 18,
            },
        },

        random_brightness: {
            type: 'object',
            name: 'Random Brightness Parameters',
            ratio: _ratio_,
            shift_value: {
                type: 'int',
                default: 32,
            },
        },

        random_perm: {
            type: 'object',
            name: 'Random Perm Parameters',
            ratio: _ratio_,
        },

        random_pad: {
            type: 'object',
            ratio: _ratio_,
            up_scale_range: {
                type: 'float-array',
                name: 'Up Scale Range',
                minnum: 2,
                maxnum: 2,
                default: [1.0, 4.0],
            },
        },

        random_resize: {  // TODO
            type: 'object',
            name: 'Random Resize Parameters',
            ratio: _ratio_,
            scale_range: {
                type: 'int-array',
                name: 'Scale Range',
                minnum: 2,
                maxnum: 2,
                default: [0.5, 1, 1],
            },
        },

        random_rotate: {
            type: 'object',
            name: 'Random Rotate Parameters',
            ratio: _ratio_,
            max_degree: {
                type: 'int',
                name: 'Max Degree',
                default: 40,
            },
        },

        random_focus_crop: {
            type: 'object',
            name: 'Random Focus Crop',
            ratio: _ratio_,
            crop_size: {
                type: 'int-array',
                name: 'Crop Size',
                minnum: 2,
                maxnum: 2,
                default: [368, 368],
            },
            center_jitter: {
                type: 'int',
                name: 'Center Jitter',
                default: 20,
            },
            allow_outside_center: {
                type: 'bool',
                name: 'Allow Outside Center',
                default: true,
            },
        },

        random_hflip: {  // TODO
            type: 'object',
            ratio: _ratio_,
            swap_pair: {
                type: 'int-array-array',
                name: 'Swap Pair Parameters',
                default: [],
            },
        },

        random_det_crop: {
            type: 'object',
            ratio: _ratio_,
        },
    },
}
