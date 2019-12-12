// @file random_perm.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:50

local lib = import 'common.libsonnet';

function(prefix) {
    local this = self,
    _id_:: prefix + '.aug_trans.random_perm',
    name: 'Random Perm Parameters',
    type: 'object',
    objs: ['ratio', 'delta'],
    ratio: lib.radio(this._id_ + '.ratio'),
}
