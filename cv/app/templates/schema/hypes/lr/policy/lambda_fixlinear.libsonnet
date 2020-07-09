// @file lambda_fixlinear.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-07-09 17:28

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        name: { en: 'Lambda Parameters', cn: self.en },
        type: 'H',
        objs: [
            _Utils.int(jid + '.fix_value',
                       'Fix Value',
                       min=1,
                       def=100),
            _Utils.int(jid + '.linear_value',
                       'Linear Value',
                       min=1,
                       def=100),
        ],
    },
}
