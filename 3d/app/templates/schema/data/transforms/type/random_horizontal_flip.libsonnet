// @file random_horizontal_flip.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-24 18:02

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        {
            type: 'H',
            objs: [
                _Utils.float(jid + '.args.p', 'Radio', def=0.5),
            ],
        },
        _Utils.checkboxphase(jid),
    ],
}
