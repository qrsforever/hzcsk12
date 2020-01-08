// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 22:26

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        _id_: jid + '.optim_method',
        name: { en: 'Type', cn: self.en },
        type: 'string-enum-trigger',
        objs: [
            {
                name: { en: 'sgd', cn: self.en },
                value: 'sgd',
                trigger: (import 'type/sgd.libsonnet').get(jid + '.sdg'),
            },
            {
                name: { en: 'adam', cn: self.en },
                value: 'adam',
                trigger: (import 'type/adam.libsonnet').get(jid + '.adam'),
            },
        ],
        default: 'adam',
    },
}
