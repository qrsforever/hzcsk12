// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-22 19:20

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'V',
        objs: [
            {
                _id_: jid + '.type',
                name: { en: 'LR Type', cn: self.en },
                type: 'string-enum-trigger',
                tips: 'set policy of decays the learning rate',
                objs: [
                    {
                        name: { en: 'StepLR', cn: self.en },
                        value: 'step',
                        trigger: (import 'type/step.libsonnet').get(jid + '.args'),
                    },
                    {
                        name: { en: 'MultiStepLR', cn: self.en },
                        value: 'multistep',
                        trigger: (import 'type/multistep.libsonnet').get(jid + '.args'),
                    },
                    {
                        name: { en: 'ReduceOnPlateau', cn: self.en },
                        value: 'reduceonplateau',
                        trigger: (import 'type/reduceonplateau.libsonnet').get(jid + '.args'),
                    },
                ],
                default: 'reduceonplateau',
            },
        ],
    },
}
