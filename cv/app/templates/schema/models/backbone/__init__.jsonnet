// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 18:13

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        _id_: '_k12.' + jid + '.backbone.enum',
        name: { en: 'Backbone', cn: self.en },
        type: 'string-enum-trigger',
        objs: [
            {
                name: { en: 'vgg', cn: self.en },
                value: 'vgg',
                trigger: (import 'vgg.libsonnet').get(jid),
            },
            {
                name: { en: 'resnet', cn: self.en },
                value: 'resnet',
                trigger: (import 'resnet.libsonnet').get(jid),
            },
        ],
        default: self.objs[0].value,
    },
}
