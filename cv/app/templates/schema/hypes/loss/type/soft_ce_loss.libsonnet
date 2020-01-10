// @file soft_ce_loss.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 21:55

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        name: { en: 'SoftCE Parameters', cn: self.en },
        type: 'H',
        objs: [
            _Utils.float('loss.loss_weights.soft_ce_loss.ce_loss', 'Weight', def=1.0),
            {
                _id_: jid + '.reduction',
                type: 'string-enum',
                name: { en: 'Reduction', cn: self.en },
                objs: [
                    { name: { en: 'batchmean', cn: self.en }, value: 'batchmean' },
                    { name: { en: 'mean', cn: self.en }, value: 'mean' },
                    { name: { en: 'sum', cn: self.en }, value: 'sum' },
                    { name: { en: 'none', cn: self.en }, value: 'none' },
                ],
                default: 'batchmean',
            },
            _Utils.float(jid + '.label_smooth', 'Label Smooth', def=0.1),
        ],
    },
}
