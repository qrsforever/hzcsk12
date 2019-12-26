// @file sentiment_analysis.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 15:01

{
    type: 'page',
    objs: [
        {
            name: { en: 'Data', cn: self.en },
            type: 'tab',
            objs: [
            ],
        },
        {
            name: { en: 'Model', cn: self.en },
            type: 'tab',
            objs: [
            ],
        },
        {
            name: { en: 'Hypes', cn: self.en },
            type: 'tab',
            objs: import '../schema/hypes/init.libsonnet',
        },
    ],
}
