// @file soft_ce_loss.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 22:05

local CE_LOSS = import 'ce_loss.libsonnet';

{
    local this = self,
    _id_:: 'loss.params.ce_loss',
    name: { en: 'CE Loss Parameters', cn: 'CE Loss 函数参数' },
    type: 'object',
    objs: ['reduction', 'label_smooth'],
    reduction: CE_LOSS.reduction {
        _id_: this._id_ + '.reduction',
        items+: [
            { name: { en: 'batchmean', cn: self.en }, value: 'batchmean' },
        ],
        default: 'batchmean',
    },
    label_smooth: {
        _id_: this._id_ + '.label_smooth',
        type: 'float',
        name: { en: 'label smooth', cn: self.en },
        default: 0.1,
    },
}
