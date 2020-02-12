// @file random_forest.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-12 17:39


local _Utils = import '../../utils/helper.libsonnet';

{
    get():: if 'classifier' == _Utils.task then
        [
            {
                type: 'H',
                objs: [
                    _Utils.int('model.random_forest.n_estimators', 'Num Estimators', def=10),
                    {
                        _id_: 'model.random_forest.criterion',
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
                    },
                    _Utils.bool('model.random_forest.bootstrap', 'Bootstrap', def=true),
                ],
            },
            {
                type: 'H',
                objs: [
                    _Utils.int('model.random_forest.n_jobs', 'Num Jobs', def=1),
                    _Utils.int('model.random_forest.max_features', 'Max Features', def=3, ddd=true),
                    _Utils.bool('model.random_forest.oob_score', 'OOB Score', def=false),
                ],
            },
            {
                type: 'H',
                objs: [
                    _Utils.int('model.random_forest.min_samples_split', 'Min S Split', def=2),
                    _Utils.int('model.random_forest.min_samples_leaf', 'Min S Leaf', def=1),
                    _Utils.bool('model.random_forest.warm_start', 'Warm Start', def=false),
                ],
            },
            {
                type: 'H',
                objs: [
                    _Utils.float('model.random_forest.min_weight_fraction_leaf', 'Min WF Leaf', def=0.0),
                    _Utils.float('model.random_forest.min_impurity_decrease', 'Min Impurity Dec', def=0.0),
                    _Utils.int('model.random_forest.verbose', 'Verbose', def=0),
                ],
            },
            {
                type: 'H',
                objs: [
                    _Utils.booltrigger('_k12.model.random_forest.max_depth',
                                       'Max Depth',
                                       def=false,
                                       trigger=[_Utils.int('model.random_forest.max_depth', 'Value', def=5, ddd=true)]),
                    _Utils.booltrigger('_k12.model.random_forest.max_leaf_nodes',
                                       'Max Leaf Nodes',
                                       def=false,
                                       trigger=[_Utils.int('model.random_forest.max_leaf_nodes', 'Value', def=15, ddd=true)]),
                    _Utils.booltrigger(
                        '_k12.model.random_forest.class_weight',
                        'Class Weight',
                        def=false,
                        trigger=[
                            {
                                _id_: 'model.random_forest.class_weight',
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
                ],
            },
        ] else if 'regressor' == _Utils.task then
        [
        ] else [],
}
