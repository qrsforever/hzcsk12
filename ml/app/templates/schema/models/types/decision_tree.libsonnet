// @file decision_tree.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-13 17:59


local _Utils = import '../../utils/helper.libsonnet';

local _C_Criterion = {
    _id_: 'model.decision_tree.criterion',
    name: { en: 'Criterion', cn: self.en },
    type: 'string-enum',
    objs: [
        {
            name: { en: 'gini', cn: self.en },
            value: 'gini',
        },
        {
            name: { en: 'entropy', cn: self.en },
            value: 'entropy',
        },
    ],
    default: 'gini',
};

local _R_Criterion = {
    _id_: 'model.decision_tree.criterion',
    name: { en: 'Criterion', cn: self.en },
    type: 'string-enum',
    objs: [
        {
            name: { en: 'mse', cn: self.en },
            value: 'mse',
        },
        {
            name: { en: 'mae', cn: self.en },
            value: 'mae',
        },
    ],
    default: 'mse',
};

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.float('model.decision_tree.max_features',
                             'Max Features',
                             def=0.3,
                             ddd=true,
                             max=0.999,
                             min=0.001),
                (
                    if 'classifier' == _Utils.task
                    then
                        _C_Criterion
                    else
                        _R_Criterion
                ),
                {
                    _id_: 'model.decision_tree.splitter',
                    name: { en: 'Splitter', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'best', cn: self.en },
                            value: 'best',
                        },
                        {
                            name: { en: 'random', cn: self.en },
                            value: 'random',
                        },
                    ],
                    default: 'random',
                },
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('model.decision_tree.min_samples_split', 'Min S Split', def=2),
                _Utils.int('model.decision_tree.min_samples_leaf', 'Min S Leaf', def=1),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.float('model.decision_tree.min_weight_fraction_leaf',
                             'Min WF Leaf',
                             def=0.0,
                             max=0.999,
                             min=0.001),
                _Utils.float('model.decision_tree.min_impurity_decrease', 'Min Impurity Dec', def=0.0),
                _Utils.bool('_k12.metrics.tree_dot', 'Display Tree', def=false),  // template put here
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.booltrigger('_k12.model.decision_tree.max_depth',
                                   'Max Depth',
                                   def=false,
                                   trigger=[_Utils.int('model.decision_tree.max_depth', 'Value', def=5, ddd=true)]),
                _Utils.booltrigger('_k12.model.decision_tree.max_leaf_nodes',
                                   'Max Leaf Nodes',
                                   def=false,
                                   trigger=[_Utils.int('model.decision_tree.max_leaf_nodes', 'Value', def=15, ddd=true)]),
                _Utils.booltrigger('_k12.model.decision_tree.random_state',
                                   'Random State',
                                   def=false,
                                   trigger=[_Utils.int('model.decision_tree.random_state', 'Value', def=1, ddd=true)]),
            ] + (
                if 'classifier' == _Utils.task then [
                    _Utils.booltrigger(
                        '_k12.model.decision_tree.class_weight',
                        'Class Weight',
                        def=false,
                        trigger=[
                            {
                                _id_: 'model.decision_tree.class_weight',
                                name: { en: 'Value', cn: self.en },
                                type: 'string-enum',
                                objs: [
                                    {
                                        name: { en: 'balanced', cn: self.en },
                                        value: 'balanced',
                                    },
                                    {
                                        name: { en: 'balanced subsample', cn: self.en },
                                        value: 'balanced_subsample',
                                    },
                                ],
                                default: 'balanced',
                            },
                        ]
                    ),
                ] else []
            ),
        },
    ],
}
