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
                _Utils.int(jid + '.num_epochs', 'epochs num', def=20, ddd=true),
                _Utils.stringenum(
                    jid + '.validation_metric',
                    'validation metric',
                    def='-loss',
                    ddd=true,
                    enums=[
                        { name: { en: '-loss', cn: self.en }, value: '-loss' },
                        { name: { en: '+accuracy', cn: self.en }, value: '+accuracy' },
                        { name: { en: '+f1-measure-overall', cn: self.en }, value: '+f1-measure-overall' },
                    ],
                    tips='validation metric to measure for whether to stop training using patience',
                ),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.booltrigger('_k12.' + jid + '.grad_norm.bool', 'grad norm', trigger=[
                    _Utils.float(
                        jid + '.grad_norm',
                        'value',
                        def=1.0,
                        min=0.001,
                        ddd=true,
                        tips='if provided, gradient norms will be rescaled to have a maximum of this value',
                    ),
                ]),
                _Utils.booltrigger('_k12.' + jid + '.grad_clipping.bool', 'grad clipping', trigger=[
                    _Utils.float(
                        jid + '.grad_clipping',
                        'value',
                        min=0.001,
                        def=1.0,
                        tips='if provided, gradients will be clipped `during the backward pass` to have an (absolute) maximum of this value.'
                    ),
                ]),
                _Utils.booltrigger('_k12.' + jid + '.patience', 'patience', trigger=[
                    _Utils.int(
                        jid + '.patience',
                        'Patience',
                        def=10,
                        ddd=true,
                        tips='number of epochs to be patient before early stopping',
                    ),
                ]),
            ],
        },
    ],
}
