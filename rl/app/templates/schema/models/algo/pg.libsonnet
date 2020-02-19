// @file pg.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-19 20:46

local _Utils = import '../../utils/helper.libsonnet';

// value_loss_coeff=1.,
// entropy_loss_coeff=0.01,
// gae_lambda=1,
// minibatches=4,
// epochs=4,
// ratio_clip=0.1,
// linear_lr_schedule=True,
// normalize_advantage=False,
{
    type: '_ignore_',
    objs: [
        {
            type: 'H',
            objs: [
                _Utils.float('algo.value_loss_coeff', 'Value Loss C', def=0.5),
                _Utils.float('algo.entropy_loss_coeff', 'Entropy Loss C', def=0.01),
                _Utils.int('algo.gae_lambda', 'Gae Lambda', def=1),
            ],
        },
        {
            _id_: '_k12.model.algo',
            name: { en: 'Type', cn: self.en },
            type: 'string-enum-trigger',
            objs: [
                {
                    name: { en: 'FF', cn: self.en },
                    value: 'ff',
                    trigger: {
                        type: '_ignore_',
                        objs: [
                            _Utils.float('model.init_log_std', 'Init Log Std', def=0.0),
                        ],
                    },
                },
                {
                    name: { en: 'LSTM', cn: self.en },
                    value: 'lstm',
                    trigger: {
                        type: '_ignore_',
                        objs: [
                            _Utils.int('model.lstm_size', 'Size', def=256, readonly=true),
                        ],
                    },
                },
            ],
            default: self.objs[0].value,
        },
    ],
}
