// @file assembel.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 08:21

[
    (import 'runner/__init__.jsonnet').get(),
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Algo', cn: self.en },
                type: '_ignore_',
                objs: (import 'algo/__init__.jsonnet').get(),
            },
            {
                name: { en: 'Replay', cn: self.en },
                type: '_ignore_',
                objs: (import 'replay/__init__.jsonnet').get(),
            },
            {
                name: { en: 'Optimizer', cn: self.en },
                type: '_ignore_',
                objs: (import 'optimizer/__init__.jsonnet').get(),
            },
        ],
    },
]
