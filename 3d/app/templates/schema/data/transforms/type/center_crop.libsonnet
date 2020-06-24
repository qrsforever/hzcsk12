// @file centercrop.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-24 18:01

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        {
            type: 'H',
            objs: [
                _Utils.intarray(jid + '.args.size', 'Size', def=[28, 28], tips='Desired output size of the crop'),
            ],
        },
        _Utils.checkboxphase(jid),
    ],
}
