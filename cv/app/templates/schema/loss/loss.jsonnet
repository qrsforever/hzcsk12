// @file loss_type.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 22:08

local lib = import 'params/common.libsonnet';

{
    local this = self,
    _id_:: 'loss',
    name: { en: 'Loss', cn: self.en },
    type: 'accordion',
    objs: [
        {
            _id_: this._id_ + '.loss_type',
            name: { en: 'loss type', cn: self.en },
            type: 'string-enum-trigger',
            objs: [
                {
                    name: { en: 'CE Loss', cn: self.en },
                    value: 'ce_loss',
                    trigger: {
                        type: 'object',
                        objs: [
                            lib.weight(this._id_ + '.loss_weights.ce_loss.ce_loss'),
                            import 'params/ce_loss.libsonnet',
                        ],
                    },
                },
                {
                    name: { en: 'SoftCE Loss', cn: self.en },
                    value: 'soft_ce_loss',
                    trigger: {
                        type: 'object',
                        objs: [
                            lib.weight(this._id_ + '.loss_weights.soft_ce_loss.soft_ce_loss'),
                            import 'params/soft_ce_loss.libsonnet',
                        ],
                    },
                },
            ],
            default: 'ce_loss',
        },
    ],
}
