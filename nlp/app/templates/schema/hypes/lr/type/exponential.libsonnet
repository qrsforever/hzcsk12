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
                         min=0.001,
                         max=0.999,
                         def=0.10,
                         tips='multiplicative factor of learning rate decay'),
        ],
    },
}
