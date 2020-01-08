// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 22:08

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'H',
        objs: [
            _Utils.float(jid + '.base_lr', 'Base LR', def=0.001),
            {
                _id_: jid + '.is_warm',
                name: { en: 'Warm up', cn: self.en },
                type: 'bool-trigger',
                objs: [
                    {
                        value: true,
                        trigger: {
                            type: '_ignore_',
                            objs: [
                                _Utils.int(jid + '.warm.warm_iters', 'Warm Iters', def=1000),
                                _Utils.float(jid + '.warm.power', 'Power', def=1.0),
                                _Utils.bool(jid + '.warm.freeze_backbone', 'Freeze Backbone', def=false),
                            ],
                        },
                    },
                    {
                        value: false,
                        trigger: {
                        },
                    },
                ],
                default: false,
            },
            {
                _id_: jid + '.lr_policy',
                name: { en: 'LR Policy', cn: self.en },
                type: 'string-enum-trigger',
                objs: [
                    {
                        name: { en: 'step', cn: self.en },
                        value: 'step',
                        trigger: (import 'policy/step.libsonnet').get(jid + '.step'),
                    },
                    {
                        name: { en: 'multi step', cn: self.en },
                        value: 'multistep',
                        trigger: (import 'policy/multistep.libsonnet').get(jid + '.multistep'),
                    },
                ],
                default: 'multistep',
            },
        ],
    },
}
