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
                objs: [
                    (import 'trainer/init.jsonnet').get(jid),
                ],
            },
            {
                name: { en: 'Optimizer', cn: self.en },
                objs: [
                    (import 'optimizer/init.jsonnet').get(jid + '.optimizer'),
                ],
            },
            {
                name: { en: 'LR Scheduler', cn: self.en },
                objs: [
                    (import 'lr/init.jsonnet').get(jid + '.learning_rate_scheduler'),
                ],
            },
        ],
    },
]
