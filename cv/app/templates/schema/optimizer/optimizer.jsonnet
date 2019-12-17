// @file optimizer.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 20:37

{
    local this = self,
    _id_:: 'solver.optim',
    name: { en: 'Optimizer', cn: self.en },
    type: 'accordion',
    objs: [
        {
            _id_: this._id_ + '.optim_method',
            name: { en: 'method', cn: self.en },
            type: 'string-enum-trigger',
            objs: [
                {
                    name: { en: 'SGD', cn: self.en },
                    value: 'sgd',
                    trigger: {
                        type: 'object',
                        objs: [
                            import 'method/sgd.libsonnet',
                        ],
                    },
                },
                {
                    name: { en: 'Adam', cn: self.en },
                    value: 'adam',
                    trigger: {
                        type: 'object',
                        objs: [
                            import 'method/adam.libsonnet',
                        ],
                    },
                },
            ],
            default: 'adam',
        },
    ],
}
