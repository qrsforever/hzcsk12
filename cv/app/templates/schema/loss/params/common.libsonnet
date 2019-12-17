// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-16 12:18

{
    weight(id): {
        _id_: id,
        type: 'float',
        name: { en: 'weight', cn: self.en },
        default: 1.0,
    },
}
