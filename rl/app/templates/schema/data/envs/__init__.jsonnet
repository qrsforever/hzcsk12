// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-17 22:13

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            name: { en: 'Phase', cn: self.en },
            type: 'navigation',
            objs: [
                {
                    name: { en: 'Train', cn: self.en },
                    type: '_ignore_',
                    objs: if _Utils.task == 'atari' then
                        (import 'atari.libsonnet').get('env', 'Train')
                    else if _Utils.task == 'classic_control' then
                        (import 'classic_control.libsonnet').get('env', 'Train')
                    else [],
                },
                {
                    name: { en: 'Evaluate', cn: self.en },
                    type: '_ignore_',
                    objs: if _Utils.task == 'atari' then
                        (import 'atari.libsonnet').get('eval_env', 'Evaluate')
                    else if _Utils.task == 'classic_control' then
                        (import 'classic_control.libsonnet').get('eval_env', 'Evaluate')
                    else [],
                },
            ],
        },
    ],
}
