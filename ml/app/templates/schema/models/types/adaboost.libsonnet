// @file adaboost.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-05 14:38

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.int('model.adaboost.n_estimators', 'Estimators', def=10, ddd=true, min=1),
                _Utils.float('mode.adaboost.learning_rate', 'LR', def=1.0, min=0.001, max=1.0),
                _Utils.booltrigger('_k12.model.adaboost.random_state',
                                   'Random State',
                                   def=false,
                                   trigger=[_Utils.int('model.adaboost.random_state', 'Value', def=100)]),
            ] + (
                if _Utils.task == 'classifier'
                then
                    [
                        _Utils.stringenum(
                            'model.adaboost.algorithm',
                            'Algo',
                            def='SAMME.R',
                            enums=[
                                {
                                    name: { en: 'SAMME', cn: self.en },
                                    value: 'SAMME',
                                },
                                {
                                    name: { en: 'SAMME.R', cn: self.en },
                                    value: 'SAMME.R',
                                },
                            ],
                            tips='SAMME.R is faster than SAMME'
                        ),
                    ]
                else [
                    _Utils.stringenum(
                        'model.adaboost.loss',
                        'Loss',
                        def='linear',
                        enums=[
                            {
                                name: { en: 'linear', cn: self.en },
                                value: 'linear',
                            },
                            {
                                name: { en: 'square', cn: self.en },
                                value: 'square',
                            },
                            {
                                name: { en: 'exponential', cn: self.en },
                                value: 'exponential',
                            },
                        ],
                        tips='updating the weights after each boosting iteration'
                    ),
                ]
            ),
        },
    ],
}
