// @file k12ai_cv.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 11:29

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
                    name: { en: 'Train', cn: self.en },
                    type: '_ignore_',
                    objs: [
                        {
                            _id_: '_k12.iframe.train',
                            type: 'iframe',
                            html: 'http://www.baidu.com',
                        },
                    ],
                },
            ],
        },
        {
            name: { en: 'Debug Output', cn: '调试输出: ' },
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
