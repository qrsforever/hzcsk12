// @file resnet.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 18:23

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            name: { en: 'resnet 18', cn: self.en },
            value: 'resnet18',
        },
        {
            name: { en: 'resnet 50', cn: self.en },
            value: 'resnet50',
        },
        {
            name: { en: 'resnet 101', cn: self.en },
            value: 'resnet101',
        },
        {
            name: { en: 'resnet 152', cn: self.en },
            value: 'resnet152',
        },
    ],
}
