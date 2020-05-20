// @file adam.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 09:05

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: {
        name: { en: 'Adam Parameters', cn: self.en },
        type: '_ignore_',
        objs: [
            {
                type: 'H',
                objs: [
                    _Utils.floatarray('optim.betas',
                                      'Betas',
                                      def=[0.9, 0.999],
                                      tips='value like [float, float], every element range [0.0, 1)'),
                    _Utils.float('optim.eps',
                                 'Eps',
                                 def=1e-8,
                                 tips='term added to the denominator to improve numerical stability'),
                ],
            },
            {
                type: 'H',
                objs: [
                    _Utils.float('optim.weight_decay',
                                 'Weight Decay',
                                 min=0.0,
                                 max=1.0,
                                 def=0.0,
                                 tips='weight decay (L2 penalty)'),
                    _Utils.bool('optim.amsgrad',
                                'Amsgrad',
                                def=false,
                                tips='whether to use the AMSGrad variant'),
                ],
            },
        ],
    },
}
