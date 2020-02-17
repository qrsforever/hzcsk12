// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 12:24

{
    get(jid):: {
        _id_: '_k12.lr_scheduler.bool',
        name: { en: 'Enable', cn: self.en },
        type: 'bool-trigger',
        objs: [
            {
                value: true,
                trigger: {
                    objs: [
                        {
                            _id_: jid + '.type',
                            name: { en: 'LR Scheduler Type', cn: self.en },
                            type: 'string-enum-trigger',
                            objs: [
                                {
                                    name: { en: 'StepLR', cn: self.en },
                                    value: 'step',
                                    trigger: (import 'type/step.libsonnet').get(jid),
                                },
                                {
                                    name: { en: 'MultiStepLR', cn: self.en },
                                    value: 'multi_step',
                                    trigger: (import 'type/multi_step.libsonnet').get(jid),
                                },
                                {
                                    name: { en: 'ExponentialLR', cn: self.en },
                                    value: 'exponential',
                                    trigger: (import 'type/exponential.libsonnet').get(jid),
                                },
                                {
                                    name: { en: 'ReduceLROnPlateau', cn: self.en },
                                    value: 'reduce_on_plateau',
                                    trigger: (import 'type/reduce_on_plateau.libsonnet').get(jid),
                                },
                            ],
                            default: self.objs[0].value,
                        },
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
}
