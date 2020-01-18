// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 12:15

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                {
                    _id_: 'network.model_name',
                    name: { en: 'Network', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: _Utils.network_name,
                            value: _Utils.network,
                        },
                    ],
                    default: self.objs[0].value,
                    readonly: true,
                },
                _Utils.bool('network.distributed', 'Distributed', def=true),
                _Utils.bool('network.resume_continue', 'Resume Continue', def=false),
            ],
        },
        {
            type: 'H',
            objs: [
                (import '../backbone/__init__.jsonnet').get(),
                _Utils.bool('network.pretrained', 'Pretrained', def=false),
                _Utils.bool('network.resume_strict', 'Resume Strict', def=false),
            ],
        },
        {
            type: 'H',
            objs: [
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
                _Utils.bool('network.syncbn', 'SyncBN', def=false),
                _Utils.bool('network.resume_val', 'Resume Validation', def=false),
            ],
        },
    ],
}