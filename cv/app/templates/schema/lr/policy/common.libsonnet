// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-12 18:00

{
    gamma(id): {
        _id_: id,
        type: 'float',
        name: { en: 'Gamma', cn: self.en },
        default: 0.10,
    },
}
