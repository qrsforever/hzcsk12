// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 00:39

(import 'networks/__init__.jsonnet').get() +
[
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Algo', cn: self.en },
                type: '_ignore_',
                objs: (import 'algo/__init__.jsonnet').get(),
            },
        ],
    },
]
