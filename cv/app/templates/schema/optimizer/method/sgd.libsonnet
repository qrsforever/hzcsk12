// @file sgd.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 11:49

local lib = import 'common.libsonnet';

{
    local this = self,
    _id_:: 'solver.optim.sgd',
    name: { en: 'SGD Parameters', cn: self.en },
    type: 'object',
    objs: ['weight_decay', 'momentum', 'nesterov'],
    weight_decay: lib.weight_decay(this._id_ + '.weight_decay'),
    momentum: {
        type: 'float',
        name: { en: 'momentum', cn: self.en },
        default: 0.9,
    },
    nesterov: {
        type: 'bool',
        name: { en: 'nesterov', cn: self.en },
        default: false,
    },
}
