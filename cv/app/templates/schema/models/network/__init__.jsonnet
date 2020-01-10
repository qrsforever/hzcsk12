// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 12:15

local _Utils = import '../../utils/helper.libsonnet';

local _NetSelector = {
    cls: [
        {
            name: { en: 'base', cn: self.en },
            value: 'base_model',
        },
        {
            name: { en: 'cls', cn: self.en },
            value: 'cls_model',
        },
        {
            name: { en: 'distill', cn: self.en },
            value: 'distill_model',
        },
        {
            name: { en: 'custom', cn: self.en },
            value: 'custom_model',
        },
    ],
};

{
    get():: [
        {
            type: 'H',
            objs: [
                (import '../backbone/__init__.jsonnet').get('network'),
                {
                    _id_: 'network.model_name',
                    name: { en: 'Network', cn: self.en },
                    type: 'string-enum',
                    objs: _NetSelector[_Utils.task],
                    default: _Utils.get_default_value(self._id_, self.objs[0].value),
                },
                {
                    _id_: 'network.norm_type',
                    name: { en: 'Norm Type', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'batch', cn: self.en },
                            value: 'batchnorm',
                        },
                        {
                            name: { en: 'sync batch', cn: self.en },
                            value: 'encsync_batchnorm',
                        },
                        {
                            name: { en: 'instance', cn: self.en },
                            value: 'instancenorm',
                        },
                    ],
                    default: self.objs[0].value,
                },
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.bool('network.pretrained', 'Pretrained', def=false),
                _Utils.bool('network.syncbn', 'SyncBN', def=false),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.bool('network.distributed', 'Distributed', def=true),
                _Utils.bool('network.gather', 'Gather', def=true),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.bool('network.resume_continue', 'Resume Continue', def=false),
                _Utils.bool('network.resume_strict', 'Resume Strict', def=false),
                _Utils.bool('network.resume_val', 'Resume Validation', def=false),
            ],
        },
        // {
        //     _id_: 'network.custom_model',
        //     name: { en: 'Custom Model', cn: self.en },
        //     type: 'bool-trigger',
        //     objs: [
        //         {
        //             value: true,
        //             trigger: _Utils.notiml(),
        //         },
        //         {
        //             value: false,
        //             trigger: {},
        //         },
        //     ],
        //     default: false,
        // },
    ],
}
