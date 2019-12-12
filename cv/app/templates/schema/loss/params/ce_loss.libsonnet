// @file ce_loss.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 22:03

{
    local this = self,
    _id_:: 'loss.params.ce_loss',
    name: { en: 'CE Loss Parameters', cn: 'CE Loss 函数参数' },
    type: 'object',
    objs: ['reduction', 'ignore_index'],
    reduction: {
        _id_: this._id_ + '.reduction',
        type: 'string-enum',
        name: { en: 'Reduction Method', cn: '简化方式' },
        items: [
            { name: { en: 'mean', cn: '平均' }, value: 'mean' },
            { name: { en: 'sum', cn: '求和' }, value: 'sum' },
            { name: { en: 'none', cn: '无' }, value: 'none' },
        ],
        default: 'mean',
    },
    ignore_index: {
        _id_: this._id_ + '.ignore_index',
        type: 'float',
        name: { en: 'Ignore Index', cn: 'Ignore Index' },
        default: -1,
    },
}
