// @file step.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 14:07

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): {
        type: 'H',
        objs: [
            _Utils.int(jid + '.step_size',
                       'Step Size',
                       def=30,
                       tips='Period of learning rate decay'),
            _Utils.float(jid + '.gamma',
                         'Gamma',
                         def=0.1,
                         tips='Multiplicative factor of learning rate decay'),
        ],
    },
}
