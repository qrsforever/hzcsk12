// @file default.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 15:52

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: [
                _Utils.int(jid + '.cuda_device', 'CUDA Device', def=0),
                _Utils.int(jid + '.num_epochs', 'Epochs Count', def=20, ddd=true),
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
                _Utils.booltrigger('_k12.' + jid + '.grad_norm.bool', 'Grad Norm', trigger=[
                    _Utils.float(jid + '.grad_norm', 'Value', def=1.0),
                ]),
                _Utils.booltrigger('_k12.' + jid + '.grad_clipping.bool', 'Grad Clipping', trigger=[
                    _Utils.float(jid + '.grad_clipping', 'Value', def=1.0),
                ]),
                _Utils.booltrigger('_k12.' + jid + '.patience', 'Patience', trigger=[
                    _Utils.int(jid + '.patience', 'Patience', def=10, ddd=true),
                ]),
            ],
        },
    ],
}
