// @file optimizer.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 20:37

{
    _id_: 'solver.optim.optim_method',
    name: { en: 'optimizer method', cn: self.en },
    type: 'string-enum',
    items: [
        {
            name: { en: 'SGD', cn: self.en },
            value: 'sgd',
            trigger: {
                type: 'object',
                objs: ['sgd'],
                sgd: import 'method/sgd.libsonnet',
            },
        },
        {
            name: { en: 'Adam', cn: self.en },
            value: 'adam',
            trigger: {
                type: 'object',
                objs: ['adam'],
                adam: import 'method/adam.libsonnet',
            },
        },
    ],
}
