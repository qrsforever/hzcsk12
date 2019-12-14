// @file optimizer.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 20:37

{
    local this = self,
    _id_:: 'solver.optim',
    name: { en: 'optimizer', cn: self.en },
    type: 'accordion',
    objs: ['optim'],
    optim: {
        _id_: this._id_ + '.optim_method',
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
    },
}
