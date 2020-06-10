// @file xgboost.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-10 21:09

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.int('model.xgboost.n_estimators', 'Estimators', def=10, ddd=true, min=1),
                _Utils.float('model.xgboost.learning_rate', 'LR', def=0.1, min=0.001, max=1.0),
                _Utils.int('model.xgboost.n_jobs', 'Jobs', def=_Utils.num_cpu, min=1),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.booltrigger('_k12.model.xgboost.gamma',
                                   'Gamma',
                                   def=false,
                                   trigger=[_Utils.float('model.xgboost.gamma', 'Gamma', def=0.3)]),
                _Utils.booltrigger('_k12.model.xgboost.max_depth',
                                   'Max Depth',
                                   def=false,
                                   trigger=[_Utils.int('model.xgboost.max_depth', 'Value', def=3)]),
                _Utils.booltrigger('_k12.model.xgboost.random_state',
                                   'Random State',
                                   def=false,
                                   trigger=[_Utils.int('model.xgboost.random_state', 'Value', def=100)]),
            ],
        },
    ] + (
        if _Utils.task == 'classifier'
        then [
            _Utils.string('model.xgboost.objective', 'objective', def='binary:logistic', readonly=true),
        ]
        else [
            _Utils.string('model.xgboost.objective', 'objective', def='reg:squarederror', readonly=true),
        ]
    ),
}
