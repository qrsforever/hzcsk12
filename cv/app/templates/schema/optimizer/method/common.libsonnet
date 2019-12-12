// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 11:50

{
    weight_decay(id): {
        _id_: id,
        type: 'float',
        name: { en: 'weight decay', cn: self.en },
        default: 0.0001,
    },
}
