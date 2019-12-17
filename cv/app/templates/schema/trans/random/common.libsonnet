// @file radio.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:17

{
    radio(id): {
        _id_: id,
        name: { en: 'ratio', cn: self.en },
        type: 'float',
        default: 0.5,
    },
    lower(id): {
        _id_: id,
        name: { en: 'lower', cn: self.en },
        type: 'float',
        default: 0.5,
    },
    upper(id): {
        _id_: id,
        name: { en: 'upper', cn: self.en },
        type: 'float',
        default: 1.5,
    },
}
