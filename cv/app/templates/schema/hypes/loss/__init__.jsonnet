// local lib = import 'params/common.libsonnet';
// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 21:56

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        _id_: jid + '.loss_type',
        name: { en: 'loss type', cn: self.en },
        type: 'string-enum-trigger',
        objs: [
            {
                name: { en: 'ce Loss', cn: self.en },
                value: 'ce_loss',
                trigger: (import 'type/ce_loss.libsonnet').get(jid + '.params.ce_loss'),
            },
            {
                name: { en: 'soft ce Loss', cn: self.en },
                value: 'soft_ce_loss',
                trigger: (import 'type/soft_ce_loss.libsonnet').get(jid + '.params.soft_ce_loss'),
            },
        ],
        default: _Utils.get_default_value(jid + '.loss_type', 'ce_loss'),
    },
}
