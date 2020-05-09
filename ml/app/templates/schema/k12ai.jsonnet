// @file k12ai_ml.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-11 23:20

{
    version: std.stripChars(importstr 'version.txt', '\n'),
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
                // {
                //     name: { en: 'Hypes', cn: self.en },
                //     type: '_ignore_',
                //     objs: import 'hypes/assemble.jsonnet',
                // },
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
                // {
                //     name: { en: 'Evaluate', cn: self.en },
                //     type: '_ignore_',
                //     objs: [
                //         {
                //             _id_: '_k12.iframe.evaluate',
                //             type: 'iframe',
                //             html: '',
                //         },
                //     ],
                // },
            ],
        },
    ],
}
