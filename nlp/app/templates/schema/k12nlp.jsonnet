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
                    objs: import 'data/assemble.jsonnet',
                },
                {
                    name: { en: 'Model', cn: self.en },
                    objs: import 'model/assemble.jsonnet',
                },
                {
                    name: { en: 'Hypes', cn: self.en },
                    objs: import 'hypes/assemble.jsonnet',
                },
            ],
        },
    ],
}
