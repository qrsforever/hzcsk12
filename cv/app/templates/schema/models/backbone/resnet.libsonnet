// @file resnet.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 18:23

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        _id_: jid + '.backbone',
        name: { en: 'Arch', cn: self.en },
        // type: 'string-enum-trigger',
        type: 'string-enum',
        objs: [
            {
                name: { en: 'ResNet_18', cn: self.en },
                value: 'resnet18',
                // trigger: {
                //     type: 'H',
                //     objs: [
                //         _Utils.string(jid + '.checkpoints_name',
                //                       'Checkpoint Name',
                //                       def='resnet18',
                //                       readonly=true),
                //         _Utils.string(jid + '.pretrained',
                //                       'Pretrained File',
                //                       def=_Utils.pretrained_path + '/resnet18-5c106cde.pth',
                //                       width=500,
                //                       readonly=true),
                //     ],
                // },
            },
            {
                name: { en: 'ResNet_34', cn: self.en },
                value: 'resnet34',
                // trigger: {
                //     type: 'H',
                //     objs: [
                //         _Utils.string(jid + '.checkpoints_name',
                //                       'Checkpoint Name',
                //                       def='resnet34',
                //                       readonly=true),
                //         _Utils.string(jid + '.pretrained',
                //                       'Pretrained File',
                //                       def=_Utils.pretrained_path + '/resnet34-333f7ec4.pth',
                //                       width=500,
                //                       readonly=true),
                //     ],
                // },
            },
            {
                name: { en: 'ResNet_50', cn: self.en },
                value: 'resnet50',
                // trigger: {
                //     type: 'H',
                //     objs: [
                //         _Utils.string(jid + '.checkpoints_name',
                //                       'Checkpoint Name',
                //                       def='resnet50',
                //                       readonly=true),
                //         _Utils.string(jid + '.pretrained',
                //                       'Pretrained File',
                //                       def=_Utils.pretrained_path + '/resnet50-19c8e357.pth',
                //                       width=500,
                //                       readonly=true),
                //     ],
                // },
            },
            {
                name: { en: 'ResNet_101', cn: self.en },
                value: 'resnet101',
                // trigger: {
                //     type: 'H',
                //     objs: [
                //         _Utils.string(jid + '.checkpoints_name',
                //                       'Checkpoint Name',
                //                       def='resnet101',
                //                       readonly=true),
                //         _Utils.string(jid + '.pretrained',
                //                       'Pretrained File',
                //                       def=_Utils.pretrained_path + '/resnet101-5d3b4d8f.pth',
                //                       width=500,
                //                       readonly=true),
                //     ],
                // },
            },
            {
                name: { en: 'ResNet_152', cn: self.en },
                value: 'resnet152',
                // trigger: {
                //     type: 'H',
                //     objs: [
                //         _Utils.string(jid + '.checkpoints_name',
                //                       'Checkpoint Name',
                //                       def='resnet152',
                //                       readonly=true),
                //         _Utils.string(jid + '.pretrained',
                //                       'Pretrained File',
                //                       def=_Utils.pretrained_path + '/resnet152-b121ed2d.pth',
                //                       width=500,
                //                       readonly=true),
                //     ],
                // },
            },
        ],
        default: 'resnet18',
    },
}
