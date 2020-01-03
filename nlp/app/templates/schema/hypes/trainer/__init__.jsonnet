// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 15:46

{
    get(jid):: {
        _id_: jid + '.type',
        name: { en: 'Trainer Type', cn: self.en },
        type: 'string-enum-trigger',
        objs: [
            {
                name: { en: 'default', cn: self.en },
                value: 'default',
                trigger: (import 'type/default.libsonnet').get(jid),
            },
            {
                name: { en: 'callback', cn: self.en },
                value: 'callback',
                trigger: {
                    objs: [],
                },
            },
        ],
        default: self.objs[0].value,
    },
}