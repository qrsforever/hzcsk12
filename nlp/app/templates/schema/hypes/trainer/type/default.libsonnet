// @file default.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 15:52

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        type: '_ignore_',
        objs: [
            {
                type: 'H',
                objs: [
                    _Utils.int(jid + '.cuda_device', 'CUDA Device', def=0),
                    _Utils.int(jid + '.num_epochs', 'Epochs Count', def=20, ddd=true),
                    _Utils.int(jid + '.patience', 'Patience', def=10, ddd=true),
                    _Utils.int(jid + '.summary_interval', 'Summary Interval', def=100, ddd=true),
                    {
                        _id_: jid + '.validation_metric',
                        name: { en: 'Validation Metric', cn: self.en },
                        type: 'string-enum',
                        objs: [
                            {
                                name: { en: '-loss', cn: self.en },
                                value: '-loss',
                            },
                            {
                                name: { en: '+accuracy', cn: self.en },
                                value: '+accuracy',
                            },
                        ],
                        default: self.objs[0].value,
                    },
                ],
            },
            {
                type: 'H',
                objs: [
                    {
                        _id_: '_k12.' + jid + 'grad_norm.bool',
                        name: { en: 'Grad Norm', cn: self.en },
                        type: 'bool-trigger',
                        objs: [
                            {
                                value: true,
                                trigger: {
                                    objs: [
                                        _Utils.float(jid + '.grad_norm', 'Value', def=1.0),
                                    ],
                                },
                            },
                            {
                                value: false,
                                trigger: {},
                            },
                        ],
                    },
                    {
                        _id_: '_k12.' + jid + 'grad_clipping.bool',
                        name: { en: 'Grad Clipping', cn: self.en },
                        type: 'bool-trigger',
                        objs: [
                            {
                                value: true,
                                trigger: {
                                    objs: [
                                        _Utils.float(jid + '.grad_clipping', 'Value', def=1.0),
                                    ],
                                },
                            },
                            {
                                value: false,
                                trigger: {},
                            },
                        ],
                    },
                ],
            },
        ],
    },
}
