// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 15:20

[
    (import 'trainer/__init__.jsonnet').get('trainer'),
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Optimizer', cn: self.en },
                type: '_ignore_',
                objs: [
                    (import 'optimizer/__init__.jsonnet').get('trainer.optimizer'),
                ],
            },
            {
                name: { en: 'LR Scheduler', cn: self.en },
                type: '_ignore_',
                objs: [
                    (import 'lr/__init__.jsonnet').get('trainer.learning_rate_scheduler'),
                ],
            },
        ],
    },
]
