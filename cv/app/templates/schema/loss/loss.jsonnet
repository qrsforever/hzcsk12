// @file loss_type.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 22:08

{
    _id_: 'loss.loss_type',
    name: { en: 'loss type', cn: self.en },
    type: 'string-enum',
    items: [
        {
            name: { en: 'ce loss', cn: self.en },
            value: 'ce_loss',
            trigger: {
                type: 'object',
                objs: ['ce_loss_weight', 'ce_loss_param'],
                ce_loss_weight: {
                    _id_: 'loss.loss_weights.ce_loss.ce_loss',
                    name: { en: 'ce loss weight', cn: self.en },
                    type: 'float',
                    default: 1.0,
                },
                ce_loss_param: import 'params/ce_loss.libsonnet',
            },
        },
        {
            name: { en: 'soft ce loss', cn: self.en },
            value: 'soft_ce_loss',
            trigger: {
                type: 'object',
                objs: ['soft_ce_loss_weight', 'soft_ce_loss_param'],
                soft_ce_loss_weight: {
                    _id_: 'loss.loss_weights.soft_ce_loss.soft_ce_loss',
                    name: { en: 'soft ce loss weight', cn: self.en },
                    type: 'float',
                    default: 1.0,
                },
                soft_ce_loss_param: import 'params/soft_ce_loss.libsonnet',
            },
        },
    ],
}
