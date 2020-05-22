// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-21 21:14

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        _id_: '_k12.' + jid + '.type',
        name: { en: 'Type', cn: self.en },
        type: 'string-enum-trigger',
        objs: [
            {
                name: { en: 'Bucket Sampler', cn: self.en },
                value: 'bucket',
                trigger: (import 'type/bucket.libsonnet').get(jid + '.batch_sampler'),
            },
            {
                name: { en: 'Basic Sampler', cn: self.en },
                value: 'basic',
                trigger: (import 'type/basic.libsonnet').get(jid, true),
            },
            {
                name: { en: 'None', cn: self.en },
                value: 'none',
                trigger: (import 'type/basic.libsonnet').get(jid, false),
            },
        ],
        default: 'bucket',
        readonly: false,
    },
}
