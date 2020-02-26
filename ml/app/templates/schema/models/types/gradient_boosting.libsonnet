// @file gradient_boosting.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-26 22:48

local _Utils = import '../../utils/helper.libsonnet';

local _C_Loss = {
    _id_: 'model.gradient_boosting.loss',
    name: { en: 'Loss', cn: self.en },
    type: 'string-enum',
    objs: [
        {
            name: { en: 'deviance', cn: self.en },
            value: 'deviance',
        },
        {
            name: { en: 'exponential', cn: self.en },
            value: 'exponential',
        },
    ],
    default: 'deviance',
};

local _R_Loss = {
    _id_: 'model.gradient_boosting.loss',
    name: { en: 'Loss', cn: self.en },
    type: 'string-enum',
    objs: [
        {
            name: { en: 'ls', cn: self.en },
            value: 'ls',
        },
        {
            name: { en: 'lad', cn: self.en },
            value: 'lad',
        },
        {
            name: { en: 'huber', cn: self.en },
            value: 'huber',
        },
        {
            name: { en: 'quantile', cn: self.en },
            value: 'quantile',
        },
    ],
    default: 'ls',
};

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.float('model.gradient_boosting.max_features', 'Max Features', def=0.3, ddd=true, max=1.0),
                (
                    if 'classifier' == _Utils.task
                    then
                        _C_Loss
                    else
                        _R_Loss
                ),
                _Utils.bool('model.gradient_boosting.warm_start', 'Warm Start', def=false),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.float('model.gradient_boosting.learning_rate', 'LR', def=0.1),
                {
                    _id_: 'model.gradient_boosting.criterion',
                    name: { en: 'Criterion', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'friedman_mse', cn: self.en },
                            value: 'friedman_mse',
                        },
                        {
                            name: { en: 'mse', cn: self.en },
                            value: 'mse',
                        },
                        {
                            name: { en: 'mae', cn: self.en },
                            value: 'mae',
                        },
                    ],
                    default: 'friedman_mse',
                },
                _Utils.int('model.gradient_boosting.max_depth', 'Max Depth', def=3),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.float('model.gradient_boosting.validation_fraction', 'Val Fraction', def=0.1, max=1.0),
                _Utils.int('model.gradient_boosting.n_iter_no_change', 'Iter No Change', def=3),
                _Utils.float('model.gradient_boosting.tol', 'Tol', def=0.0001, max=1.0),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.float('model.gradient_boosting.subsample', 'Subsample', max=1.0, def=1.0),
                _Utils.int('model.gradient_boosting.min_samples_split', 'Min S Split', def=2),
                _Utils.int('model.gradient_boosting.min_samples_leaf', 'Min S Leaf', def=1),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('model.gradient_boosting.n_estimators', 'Num Estimators', def=100),
                _Utils.float('model.gradient_boosting.min_weight_fraction_leaf', 'Min WF Leaf', def=0.0, max=0.999999),
                _Utils.float('model.gradient_boosting.min_impurity_decrease', 'Min Impurity Dec', def=0.0),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.booltrigger('_k12.model.gradient_boosting.max_leaf_nodes',
                                   'Max Leaf Nodes',
                                   def=false,
                                   trigger=[_Utils.int('model.gradient_boosting.max_leaf_nodes', 'Value', def=15, ddd=true)]),
                _Utils.booltrigger('_k12.model.gradient_boosting.random_state',
                                   'Random State',
                                   def=false,
                                   trigger=[_Utils.int('model.gradient_boosting.random_state', 'Value', def=42, ddd=true)]),
                _Utils.booltrigger('_k12.model.gradient_boosting.verbose',
                                   'Verbose',
                                   def=false,
                                   trigger=[_Utils.int('model.gradient_boosting.verbose', 'Value', def=0)]),
            ],
        },
    ],
}
