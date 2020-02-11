// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-11 23:33


local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.string('model.name', 'Type', def=_Utils.network, readonly=true),
            {
                _id_: 'model.args.kernel',
                name: { en: 'Kernel', cn: self.en },
                type: 'string-enum',
                objs: [
                    {
                        name: { en: 'rbf', cn: self.en },
                        value: 'rbf',
                    },
                ],
                default: 'rbf',
            },
        ],
    },
    {
        type: 'H',
        objs: [
            {
                _id_: 'model.args.kernel',
                name: { en: 'Kernel', cn: self.en },
                type: 'string-enum',
                objs: [
                    {
                        name: { en: 'rbf', cn: self.en },
                        value: 'rbf',
                    },
                ],
                default: 'rbf',
            },
            _Utils.float('model.args.C', 'C', def=1.0),
            _Utils.int('model.args.degree', 'degree', def=3),
            _Utils.int('model.args.max_iter', 'degree', def=1000),
            _Utils.int('model.args.random_state', 'random state', def=1),
            _Utils.float('model.args.gamma', 'gamma', def=0.2),
            _Utils.float('model.args.coef0', 'coef', def=0.0),
            _Utils.bool('model.args.probability', 'probability', def=false),
            _Utils.bool('model.args.shrinking', 'shrinking', def=true),
            _Utils.float('model.args.tol', 'tol', def=0.001),
            _Utils.float('model.args.cache_size', 'cache_size', def=200.0),
            {
                _id_: 'model.args.decision_function_shape',
                name: { en: 'Decision', cn: self.en },
                type: 'string-enum',
                objs: [
                    {
                        name: { en: 'ovr', cn: self.en },
                        value: 'ovr',
                    },
                ],
                default: 'ovr',
            },
        ],
    },
]
