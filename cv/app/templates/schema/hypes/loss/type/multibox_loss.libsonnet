// @file multibox_loss.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-13 22:43

local _Utils = import '../../../utils/helper.libsonnet';

{
    get():: {
        name: { en: 'MultiBox Loss Parameters', cn: self.en },
        type: 'H',
        objs: [
            _Utils.float('loss.loss_weights.multibox_loss.multibox_loss', 'Weight', def=1.0),
        ],
    },
}
