// @file random_saturation.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:52

local _ratio_ = import 'radio.libsonnet';

{
    object(prefix):: {
        local this = self,
        _id_: prefix + '.aug_trans.random_saturation',
        type: 'object',
        name: 'Random Saturation Parameters',
        objs: ['ratio', 'lower', 'upper'],
        ratio: _ratio_ { _id_: this._id_ + '.ratio' },
        lower: {
            _id_: this._id_ + '.lower',
            type: 'float',
            default: 0.5,
        },
        upper: {
            _id_: this._id_ + '.upper',
            type: 'float',
            default: 1.5,
        },
    },
}
