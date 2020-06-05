// @file regressor.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-27 17:15

local _Utils = import '../utils/helper.libsonnet';

local _MultiOuptAgg(method) = {
    _id_: 'metrics.' + method + '.multioutput',
    name: { en: 'Agg', cn: self.en },
    type: 'string-enum',
    objs: [
        {
            name: { en: 'raw', cn: self.en },
            value: 'raw_values',
        },
        {
            name: { en: 'uniform', cn: self.en },
            value: 'uniform_average',
        },
        {
            name: { en: 'variance', cn: self.en },
            value: 'variance_weighted',
        },
    ],
    default: _Utils.get_default_value(self._id_, 'uniform_average'),
    tips: 'Defines aggregating of multiple output values',
};

[
    {
        type: 'H',
        objs: [
            _Utils.booltrigger('_k12.metrics.r2',
                               'R2 Score',
                               def=true,
                               trigger=[_MultiOuptAgg('r2')],
                               readonly=true),
            _Utils.booltrigger('_k12.metrics.mae',
                               'MAE',
                               def=false,
                               trigger=[_MultiOuptAgg('mae')]),
            _Utils.booltrigger('_k12.metrics.mse',
                               'MSE',
                               def=false,
                               trigger=[_MultiOuptAgg('mse')]),
        ],
    },
    {
        type: 'H',
        objs: [
            _Utils.booltrigger('_k12.metrics.msle',
                               'MSLE',
                               def=false,
                               trigger=[_MultiOuptAgg('msle')]),
            _Utils.booltrigger('_k12.metrics.mdae',
                               'MDAE',
                               def=false,
                               trigger=[_MultiOuptAgg('mdae')]),
            _Utils.booltrigger('_k12.metrics.evs',
                               'EVS',
                               def=false,
                               trigger=[_MultiOuptAgg('evs')]),
        ],
    },
] + (import 'common.libsonnet').get()
