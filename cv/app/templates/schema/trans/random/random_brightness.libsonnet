// @file random_brightness.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:02

local _ratio_ = import 'radio.libsonnet';

{
    object(prefix):: {
        local this = self,
        _id_: prefix + '.aug_trans.random_brightness',
        type: 'object',
        name: 'Random Brightness Parameters',
        objs: ['ratio', 'shift_value'],
        ratio: _ratio_ { _id_: this._id_ + '.ratio' },
        shift_value: {
            _id_: this._id_ + '.shift_value',
            type: 'int',
            default: 32,
        },
    },
}
