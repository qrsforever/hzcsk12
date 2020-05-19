// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 22:08

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'V',
        objs: [
            _Utils.float(jid + '.base_lr',
                         'Base LR',
                         min=0.0,
                         max=1.0,
                         def=0.01,
                         tips='learning rate'),
            {
                _id_: jid + '.lr_policy',
                name: { en: 'LR Policy', cn: self.en },
                type: 'string-enum-trigger',
                tips: 'set policy of decays the learning rate',
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
            {
                _id_: jid + '.is_warm',
                name: { en: 'Warm up', cn: self.en },
                type: 'bool-trigger',
                objs: [
                    {
                        value: true,
                        trigger: {
                            type: 'H',
                            objs: [
                                _Utils.int(jid + '.warm.warm_iters',
                                           'Warm Iters',
                                           def=1000,
                                           tips='warmup is working within the iters count'),
                                _Utils.int(jid + '.warm.power',
                                           'Power',
                                           min=1,
                                           max=10,
                                           def=1,
                                           tips='set learning rate power value'),
                                _Utils.bool(jid + '.warm.freeze_backbone',
                                            'Freeze',
                                            width=250,
                                            def=false,
                                            tips='set learning rate to 0 forcely within warm iters'),
                            ],
                        },
                    },
                    {
                        value: false,
                        trigger: {},
                    },
                ],
                default: false,
            },
        ],
    },
}
