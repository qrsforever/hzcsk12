// @file random_contrast.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:38

local lib = import 'common.libsonnet';

function(prefix) {
    local this = self,
    _id_:: prefix + '.aug_trans.random_contrast',
    name: 'Random Contrast Parameters',
    type: 'object',
    objs: ['ratio', 'lower', 'upper'],
    ratio: lib.radio(this._id_ + '.ratio'),
    lower: lib.lower(this._id_ + '.lower'),
    upper: lib.upper(this._id_ + '.upper'),
}
