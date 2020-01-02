// @file default.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 15:52

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): {
        objs: [
            {
                type: 'H',
                objs: [
                    _Utils.int(jid + '.cuda_device', 'CUDA Device', def=0, readonly=true),
                    _Utils.int(jid + '.num_epochs', 'Epochs Num', def=20),
                    _Utils.int(jid + '.patience', 'Patience', def=10),
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
                        name: { en: 'Enable Grad Norm', cn: self.en },
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
                        name: { en: 'Enable Grad Clipping', cn: self.en },
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
