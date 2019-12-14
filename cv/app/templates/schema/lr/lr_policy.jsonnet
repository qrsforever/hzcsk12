// @file lr_policy.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 19:47

{
    _id_: 'lr.lr_policy',
    name: { en: 'LR Policy', cn: self.en },
    type: 'string-enum-trigger',
    objs: [
        {
            name: { en: 'Step', cn: self.en },
            value: 'step',
            trigger: import 'policy/step.libsonnet',
        },
        {
            name: { en: 'MultiStep', cn: self.en },
            value: 'multistep',
            trigger: import 'policy/multistep.libsonnet',
        },
    ],
    default: 'multistep',
}
