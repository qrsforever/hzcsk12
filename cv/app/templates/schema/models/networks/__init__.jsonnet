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
                _Utils.string('network.model_name', 'Network', def=_Utils.network, readonly=true),
                _Utils.bool('network.distributed', 'Distributed', def=false, readonly=true),
                _Utils.bool('network.resume_continue',
                            'Resume Continue',
                            def=false,
                            tips='continue with the last training'),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.string('network.backbone', 'Backbone', def=_Utils.backbone, readonly=true),
                _Utils.bool('network.pretrained',
                            'Pretrained',
                            def=false,
                            tips='if true using the pretrained models weights, not support custom model'),
                _Utils.bool('network.resume_strict', 'Resume Strict', def=false, readonly=true),
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
                    readonly: true,
                },
                _Utils.bool('network.syncbn', 'SyncBN', def=false, tips='whether to sync BN'),
                _Utils.bool('network.resume_val',
                            'Resume Validation',
                            def=false,
                            tips='continue with the last training, first execute the val set'),
            ],
        },
    ],
}
