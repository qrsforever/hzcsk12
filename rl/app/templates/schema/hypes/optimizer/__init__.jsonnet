// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 22:26

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: '_ignore_',
            objs: [
                {
                    _id_: '_k12.algo.optimcls',
                    name: { en: 'Optimizer Type', cn: self.en },
                    type: 'string-enum-trigger',
                    objs: [
                        {
                            name: { en: 'adam', cn: self.en },
                            value: 'adam',
                            trigger: (import 'adam.libsonnet').get(),
                        },
                    ],
                    default: self.objs[0].value,
                    readonly: true,
                },
            ],
        },
    ],
}