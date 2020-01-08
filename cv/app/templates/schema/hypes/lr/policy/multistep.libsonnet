// @file multistep.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 22:20

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        name: { en: 'MultiStep Parameters', cn: self.en },
        type: 'H',
        objs: [
            _Utils.float(jid + '.gamma', 'Gamma', def=0.10),
            _Utils.intarray(jid + '.stepvalue', 'Multi Size', def=[90, 120]),
        ],
    },
}
