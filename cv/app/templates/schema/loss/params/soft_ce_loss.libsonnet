// @file soft_ce_loss.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 22:05

{
    local this = self,
    _id_:: 'loss.params.ce_loss',
    name: { en: 'SoftCE Parameters', cn: 'CE Loss 函数参数' },
    type: 'object',
    objs: [
        {
            _id_: this._id_ + '.reduction',
            type: 'string-enum',
            name: { en: 'reduction', cn: '简化方式' },
            objs: [
                { name: { en: 'mean', cn: '平均' }, value: 'mean' },
                { name: { en: 'sum', cn: '求和' }, value: 'sum' },
                { name: { en: 'none', cn: '无' }, value: 'none' },
                { name: { en: 'batchmean', cn: self.en }, value: 'batchmean' },
            ],
            default: 'batchmean',
        },
        {
            _id_: this._id_ + '.label_smooth',
            type: 'float',
            name: { en: 'label smooth', cn: self.en },
            default: 0.1,
        },
    ],
}
