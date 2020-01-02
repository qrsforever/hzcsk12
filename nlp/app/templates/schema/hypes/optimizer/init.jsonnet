// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 12:24

{
    get(jid): {
        _id_: jid + '.type',
        name: { en: 'Optimizer Type', cn: self.en },
        type: 'string-enum-trigger',
        objs: [
            {
                name: { en: 'adam', cn: self.en },
                value: 'adam',
                trigger: (import 'type/adam.libsonnet').get(jid),
            },
            {
                name: { en: 'sgd', cn: self.en },
                value: 'sgd',
                trigger: (import 'type/sgd.libsonnet').get(jid),
            },
        ],
        default: self.objs[0].value,
    },
}
