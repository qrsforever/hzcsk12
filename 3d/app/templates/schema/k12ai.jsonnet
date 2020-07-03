// @file k12ai.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-22 17:59

local _Utils = import 'utils/helper.libsonnet';

{
    version: std.stripChars(importstr 'version.txt', '\n'),
    type: 'page',
    objs: [
        {
            type: 'tab',
            objs: [
                {
                    name: { en: 'Data', cn: '数据' },
                    type: '_ignore_',
                    objs: import 'data/assemble.jsonnet',
                },
                {
                    name: { en: 'Model', cn: '模型' },
                    type: '_ignore_',
                    objs: import 'models/assemble.jsonnet',
                },
                {
                    name: { en: 'Hypes', cn: '超参' },
                    type: '_ignore_',
                    objs: import 'hypes/assemble.jsonnet',
                },
                {
                    name: { en: 'Metrics', cn: '指标' },
                    type: '_ignore_',
                    objs: import 'metrics/assemble.jsonnet',
                },
                {
                    name: { en: 'Train', cn: '训练' },
                    type: '_ignore_',
                    objs: [
                        {
                            _id_: '_k12.iframe.train',
                            type: 'iframe',
                            html: '',
                        },
                    ],
                },
            ],
        },
    ],
}
