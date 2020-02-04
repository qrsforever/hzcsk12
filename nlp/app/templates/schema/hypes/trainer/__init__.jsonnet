// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 15:46

local _Utils = import '../../utils/helper.libsonnet';

[
    {
        _id_: '_k12.' + 'trainer.type',
        name: { en: 'Trainer Type', cn: self.en },
        type: 'string-enum-trigger',
        objs: [
            {
                name: { en: 'default', cn: self.en },
                value: 'default',
                trigger: { objs: [] },
            },
            {
                name: { en: 'callback', cn: self.en },
                value: 'callback',
                trigger: { objs: [] },
            },
        ],
        readonly: true,
        default: self.objs[0].value,
    },
] + (import 'type/default.libsonnet').get('trainer')
