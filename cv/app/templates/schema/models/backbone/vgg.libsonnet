// @file vgg.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 18:19

{
    get():: [
        {
            name: { en: 'vgg 11', cn: self.en },
            value: 'vgg11',
        },
        {
            name: { en: 'vgg 16', cn: self.en },
            value: 'vgg16',
        },
        {
            name: { en: 'vgg 19', cn: self.en },
            value: 'vgg19',
        },
        {
            name: { en: 'vgg 16 bn', cn: self.en },
            value: 'vgg16_bn',
        },
        {
            name: { en: 'vgg 19 bn', cn: self.en },
            value: 'vgg19_bn',
        },
    ],
}
