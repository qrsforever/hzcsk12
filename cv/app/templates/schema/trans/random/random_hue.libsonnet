// @file random_hue.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:21

local _ratio_ = import 'radio.libsonnet';

{
    object(prefix):: {
        local this = self,
        _id_: prefix + '.aug_trans.random_hue',
        type: 'object',
        name: 'Random Hue Parameters',
        object: ['ratio', 'delta'],
        ratio: _ratio_ { _id_: this._id_ + '.ratio' },
        delta: {
            _id_: this._id_ + '.delta',
            type: 'int',
            default: 18,
        },
    },
}
