// @file random_hue.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:21

local lib = import 'common.libsonnet';

function(prefix) {
    local this = self,
    _id_:: prefix + '.aug_trans.random_hue',
    name: 'Random Hue Parameters',
    type: 'object',
    object: ['ratio', 'delta'],
    ratio: lib.radio(this._id_ + '.ratio'),
    delta: {
        _id_: this._id_ + '.delta',
        type: 'int',
        name: { en: 'delta', cn: self.en },
        default: 18,
    },
}
