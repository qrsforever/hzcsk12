// @file adam.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 11:59

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'H',
        objs: [
            _Utils.floatarray(jid + '.betas',
                              'Betas',
                              def=[0.9, 0.999]),
            _Utils.float(jid + '.eps',
                         'Eps',
                         def=1e-8),
        ] + (import 'common.libsonnet').get(jid),
    },
}
