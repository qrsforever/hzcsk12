// @file random_perm.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:50

local _ratio_ = import 'radio.libsonnet';

{
    object(prefix):: {
        local this = self,
        _id_: prefix + '.aug_trans.random_perm',
        type: 'object',
        name: 'Random Perm Parameters',
        object: ['ratio', 'delta'],
        ratio: _ratio_ { _id_: this._id_ + '.ratio' },
    },
}
