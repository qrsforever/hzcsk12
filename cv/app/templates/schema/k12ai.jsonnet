// @file k12ai_cv.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 11:29

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
            ] + (
                if std.startsWith(_Utils.network, 'custom_') then [
                    {
                        name: { en: 'Custom', cn: '自定义' },
                        type: '_ignore_',
                        objs: [
                            {
                                _id_: 'network.net_def',
                                type: 'iframe',
                                html: '',
                                width: 800,
                                height: 400,
                            },
                        ],
                    },
                ] else []
            ) + [
                {
                    name: { en: 'Hypes', cn: '超参' },
                    type: '_ignore_',
                    objs: import 'hypes/assemble.jsonnet',
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
                {
                    name: { en: 'Evaluate', cn: '评估' },
                    type: '_ignore_',
                    objs: [
                        {
                            _id_: '_k12.iframe.evaluate',
                            type: 'iframe',
                            html: '',
                        },
                    ],
                },
            ],
        },
    ],
}
