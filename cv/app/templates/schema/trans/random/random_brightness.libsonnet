// @file random_brightness.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:02

local lib = import 'common.libsonnet';

function(prefix) {
    local this = self,
    _id_:: prefix + '.aug_trans.random_brightness',
    name: 'Random Brightness Parameters',
    type: 'object',
    objs: ['ratio', 'shift_value'],
    ratio: lib.radio(this._id_ + '.ratio'),
    shift_value: {
        _id_: this._id_ + '.shift_value',
        type: 'int',
        default: 32,
    },
}
