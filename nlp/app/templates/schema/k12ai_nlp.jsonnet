// @file k12nlp.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 20:06

{
    type: 'page',
    objs: [
        {
            type: 'tab',
            objs: [
                {
                    name: { en: 'Data', cn: self.en },
                    type: '_ignore_',
                    objs: import 'data/assemble.jsonnet',
                },
                {
                    name: { en: 'Model', cn: self.en },
                    type: '_ignore_',
                    objs: import 'models/assemble.jsonnet',
                },
                {
                    name: { en: 'Hypes', cn: self.en },
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
        {
            name: { en: 'Debug:', cn: '调试: ' },
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
