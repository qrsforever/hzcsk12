// @file logistic.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-24 23:00

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.int('model.logistic.max_iter', 'Max Iter', def=100),
                {
                    _id_: 'model.logistic.penalty',
                    name: { en: 'Penalty', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'l1', cn: self.en },
                            value: 'l1',
                        },
                        {
                            name: { en: 'l2', cn: self.en },
                            value: 'l2',
                        },
                        {
                            name: { en: 'elasticnet', cn: self.en },
                            value: 'elasticnet',
                        },
                    ],
                    default: _Utils.get_default_value(self._id_, 'l2'),
                },
                _Utils.bool('model.logistic.dual', 'Dual', def=false, tips='only impl for l2 penalty'),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('model.logistic.n_jobs', 'Num Jobs', def=1),
                {
                    _id_: 'model.logistic.multi_class',
                    name: { en: 'Multi Class', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'ovr', cn: self.en },
                            value: 'ovr',
                        },
                        {
                            name: { en: 'multinomial', cn: self.en },
                            value: 'multinomial',
                        },
                        {
                            name: { en: 'auto', cn: self.en },
                            value: 'auto',
                        },
                    ],
                    default: _Utils.get_default_value(self._id_, 'ovr'),
                },
                _Utils.bool('model.logistic.warm_start', 'Warm Start', def=false),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.float('model.logistic.tol', 'Tolerance', def=0.001, max=0.999999),
                {
                    _id_: 'model.logistic.solver',
                    name: { en: 'Solver', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'liblinear', cn: self.en },
                            value: 'liblinear',
                        },
                        {
                            name: { en: 'newton-cg', cn: self.en },
                            value: 'newton-cg',
                        },
                        {
                            name: { en: 'lbfgs', cn: self.en },
                            value: 'lbfgs',
                        },
                        {
                            name: { en: 'sag', cn: self.en },
                            value: 'sag',
                        },
                        {
                            name: { en: 'saga', cn: self.en },
                            value: 'saga',
                        },
                    ],
                    default: _Utils.get_default_value(self._id_, 'liblinear'),
                },
                _Utils.bool('model.logistic.fit_intercept', 'Intercept', def=true),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.float('model.logistic.C', 'C', min=0.001, def=1.0),
                _Utils.int('model.logistic.verbose', 'Verbose', def=0),
                _Utils.float('model.logistic.intercept_scaling', 'Intercept Scaling', def=1),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.booltrigger('_k12.model.logistic.random_state',
                                   'Random State',
                                   def=false,
                                   trigger=[_Utils.int('model.logistic.random_state', 'Value', def=1, ddd=true)]),
                _Utils.booltrigger('_k12.model.logistic.l1_ratio',
                                   'L1 Ratio',
                                   def=false,
                                   trigger=[_Utils.float('model.logistic.l1_ratio', 'Value', def=0, min=0, max=1)]),
            ],
        },
    ],
}
