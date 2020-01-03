// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 15:20

[
    {
        local jid = 'trainer',
        type: 'accordion',
        objs: [
            {
                name: { en: 'Trainer', cn: self.en },
                type: '_ignore_',
                objs: [
                    (import 'trainer/__init__.jsonnet').get(jid),
                ],
            },
            {
                name: { en: 'Optimizer', cn: self.en },
                type: '_ignore_',
                objs: [
                    (import 'optimizer/__init__.jsonnet').get(jid + '.optimizer'),
                ],
            },
            {
                name: { en: 'LR Scheduler', cn: self.en },
                type: '_ignore_',
                objs: [
                    (import 'lr/__init__.jsonnet').get(jid + '.learning_rate_scheduler'),
                ],
            },
        ],
    },
]
