// @file svr.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-12 19:14


local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                {
                    _id_: 'model.svr.kernel',
                    name: { en: 'Kernel', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'linear', cn: self.en },
                            value: 'linear',
                        },
                        {
                            name: { en: 'rbf', cn: self.en },
                            value: 'rbf',
                        },
                        {
                            name: { en: 'poly', cn: self.en },
                            value: 'poly',
                        },
                        {
                            name: { en: 'sigmoid', cn: self.en },
                            value: 'sigmoid',
                        },
                        {
                            name: { en: 'precomputed', cn: self.en },
                            value: 'precomputed',
                        },
                    ],
                    default: 'rbf',
                },
                {
                    _id_: 'model.svr.gamma',
                    name: { en: 'Gamma', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'auto', cn: self.en },
                            value: 'auto',
                        },
                        {
                            name: { en: 'scale', cn: self.en },
                            value: 'scale',
                        },
                    ],
                    default: 'auto',
                },
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('model.svr.max_iter', 'Max Iter', def=1000),
                _Utils.int('model.svr.degree', 'Degree', def=3),
                _Utils.bool('model.svr.shrinking', 'Shrinking', def=true),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.float('model.svr.C', 'C', def=1.0),
                _Utils.float('model.svr.coef0', 'Coef0', def=0.0),
                _Utils.bool('model.svr.verbose', 'Verbose', def=false),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.float('model.svr.tol', 'Tolerance', def=0.001),
                _Utils.float('model.svr.cache_size', 'Cache Size', def=200.0),
                _Utils.float('model.svr.epsilon', 'Epsilon', def=0.1),
            ],
        },
    ],
}
