// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-17 22:13

{
    get():: [
        {
            name: { en: 'Phase', cn: self.en },
            type: 'navigation',
            objs: [
                {
                    name: { en: 'Train', cn: self.en },
                    type: '_ignore_',
                    objs: (import 'atari.libsonnet').get('env', 'Train'),
                },
                {
                    name: { en: 'Evaluate', cn: self.en },
                    type: '_ignore_',
                    objs: (import 'atari.libsonnet').get('eval_env', 'Evaluate'),
                },
            ],
        },
    ],
}
