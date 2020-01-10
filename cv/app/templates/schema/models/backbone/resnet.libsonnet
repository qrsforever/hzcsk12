// @file resnet.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 18:23

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        type: '_ignore_',
        objs: [
            {
                _id_: jid + '.backbone',
                name: { en: 'Arch', cn: self.en },
                type: 'string-enum',
                objs: [
                    {
                        name: { en: 'resnet_18', cn: self.en },
                        value: 'resnet18',
                    },
                    {
                        name: { en: 'resnet_34', cn: self.en },
                        value: 'resnet34',
                    },
                    {
                        name: { en: 'resnet_50', cn: self.en },
                        value: 'resnet50',
                    },
                    {
                        name: { en: 'resnet_101', cn: self.en },
                        value: 'resnet101',
                    },
                    {
                        name: { en: 'resnet_152', cn: self.en },
                        value: 'resnet152',
                    },
                ],
                default: 'resnet18',
            },
        ],
    },
}
