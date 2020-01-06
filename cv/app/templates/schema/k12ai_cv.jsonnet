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
            ],
        },
        {
            name: { en: 'Debug Output', cn: '调试输出: ' },
            type: 'output',
            objs: [],
        },
    ],
}
