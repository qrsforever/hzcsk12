// @file exponential.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 14:06

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'H',
        objs: [
            _Utils.float(jid + '.gamma',
                         'Gamma',
                         def=0.1,
                         tips='Multiplicative factor of learning rate decay'),
        ],
    },
}
