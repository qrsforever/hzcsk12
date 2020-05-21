// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-21 21:14

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        _id_: jid + '.type',
        name: { en: 'Type', cn: self.en },
        type: 'string-enum-trigger',
        objs: [
            {
                name: { en: 'bucket', cn: self.en },
                value: 'bucket',
                trigger: (import 'type/bucket.libsonnet').get(jid),
            },
        ],
        default: 'bucket',
        readonly: false,
    },
}
