// @file k12ai_rl.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-17 14:51

{
    type: 'page',
    objs: [
        {
            type: 'tab',
            objs: [
                {
                    name: { en: 'Data', cn: '数据' },
                    objs: import 'data/assemble.jsonnet',
                },
                {
                    name: { en: 'Model', cn: '模型' },
                    objs: import 'models/assemble.jsonnet',
                },
                {
                    name: { en: 'Hypes', cn: '超参' },
                    objs: import 'hypes/assemble.jsonnet',
                },
                {
                    name: { en: 'Train', cn: '训练' },
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
        {
            name: { en: 'Debug: ', cn: '调试: ' },
            type: 'output',
            objs: [
                { value: 'print', name: 'Print' },
                { value: 'kv', name: 'Key-Value(changed)' },
                { value: 'json', name: 'Json(changed)' },
                { value: 'kvs', name: 'Key-Value(all)' },
                { value: 'jsons', name: 'Json(all)' },
            ],
        },
    ],
}
