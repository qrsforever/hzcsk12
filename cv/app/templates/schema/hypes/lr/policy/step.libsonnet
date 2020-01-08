// @file step.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 22:17

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        name: { en: 'Step Parameters', cn: self.en },
        type: 'H',
        objs: [
            _Utils.float(jid + '.gamma', 'Gamma', def=0.10),
            _Utils.int(jid + '.step_size', 'Step Size', def=50),
        ],
    },
}
