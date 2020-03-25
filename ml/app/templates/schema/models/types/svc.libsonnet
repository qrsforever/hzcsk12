// @file svc.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-12 12:16


local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                {
                    _id_: 'model.svc.kernel',
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
                    _id_: 'model.svc.gamma',
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
                    tips: 'Kernel coefficient',
                    default: 'auto',
                },
                {
                    _id_: 'model.svc.decision_function_shape',
                    name: { en: 'Decision', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'ovr', cn: self.en },
                            value: 'ovr',
                        },
                        {
                            name: { en: 'ovo', cn: self.en },
                            value: 'ovo',
                        },
                    ],
                    default: 'ovr',
                },
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('model.svc.max_iter', 'Max Iter', def=1000),
                _Utils.int('model.svc.degree', 'Degree', def=3),
                _Utils.bool('model.svc.shrinking', 'Shrinking', def=true),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.float('model.svc.C', 'C', def=1.0, tips='Penalty parameter C of the error term, larger is better, but maybe over fit'),
                _Utils.float('model.svc.coef0', 'Coef0', def=0.0),
                _Utils.bool('model.svc.verbose', 'Verbose', def=false),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.float('model.svc.tol', 'Tolerance', def=0.001, max=0.999999),
                _Utils.float('model.svc.cache_size', 'Cache Size', def=200.0),
                _Utils.bool('model.svc.probability', 'Probability', def=false),
            ],
        },
    ],
}
