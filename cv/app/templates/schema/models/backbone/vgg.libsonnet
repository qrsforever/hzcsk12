// @file vgg.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 18:19

{
    get(jid):: {
        _id_: jid + '.backbone',
        name: { en: 'Arch', cn: self.en },
        type: 'string-enum',
        objs: [
            {
                name: { en: 'VGG_11', cn: self.en },
                value: 'vgg11',
            },
            {
                name: { en: 'VGG_13', cn: self.en },
                value: 'vgg13',
            },
            {
                name: { en: 'VGG_16', cn: self.en },
                value: 'vgg16',
            },
            {
                name: { en: 'VGG_19', cn: self.en },
                value: 'vgg19',
            },
            {
                name: { en: 'VGG_13_bn', cn: self.en },
                value: 'vgg13_bn',
            },
            {
                name: { en: 'VGG_16_bn', cn: self.en },
                value: 'vgg16_bn',
            },
            {
                name: { en: 'VGG_19_bn', cn: self.en },
                value: 'vgg19_bn',
            },
        ],
        default: 'vgg16',
    },
}
