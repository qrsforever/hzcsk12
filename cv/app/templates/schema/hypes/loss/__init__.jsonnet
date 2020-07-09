// local lib = import 'params/common.libsonnet';
// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 21:56

local _Utils = import '../../utils/helper.libsonnet';

local _LossSelect = {

};

{
    get(jid):: {
        _id_: jid + '.loss_type',
        name: { en: 'loss type', cn: self.en },
        type: 'string-enum-trigger',
        objs: if _Utils.task == 'cls' then [
            {
                name: { en: 'ce loss', cn: self.en },
                value: 'ce_loss',
                trigger: (import 'type/ce_loss.libsonnet').get(jid + '.params.ce_loss'),
            },
            {
                name: { en: 'soft ce loss', cn: self.en },
                value: 'soft_ce_loss',
                trigger: (import 'type/soft_ce_loss.libsonnet').get(jid + '.params.soft_ce_loss'),
            },
        ] else if _Utils.task == 'det' then [
            {
                name: { en: 'multibox loss', cn: self.en },
                value: 'multibox_loss',
                trigger: (import 'type/multibox_loss.libsonnet').get(),
            },
        ] else if _Utils.task == 'gan' then [
            {
                name: { en: 'gan loss', cn: self.en },
                value: 'gan_loss',
                trigger: {
                    _id_: 'loss.params.gan_loss.gan_mode',
                    name: { en: 'type', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'vanilla', cn: self.en },
                            value: 'vanilla',
                        },
                        {
                            name: { en: 'lsgan', cn: self.en },
                            value: 'lsgan',
                        },
                        {
                            name: { en: 'wgangp', cn: self.en },
                            value: 'wgangp',
                        },
                    ],
                    default: self.objs[0].value,
                },
            },
        ],
        default: _Utils.get_default_value(self._id_, self.objs[0].value),
        readonly: true,
    },
}
