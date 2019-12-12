// @file lr_policy.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 19:47

{
    _id_: 'lr.lr_policy',
    name: { en: 'LR Policy', cn: self.en },
    type: 'string-enum',
    items: [
        {
            name: { en: 'Step', cn: self.en },
            value: 'step',
            trigger: {
                type: 'object',
                objs: ['step'],
                step: import 'policy/step.libsonnet',
            },
        },
        {
            name: { en: 'MultiStep', cn: self.en },
            value: 'multistep',
            trigger: {
                type: 'object',
                objs: ['multistep'],
                step: import 'policy/multistep.libsonnet',
            },
        },
    ],
    default: 'multistep',
}
