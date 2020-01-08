// @file random_crop.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 19:45

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.radio', 'radio', def=0.5),
        _Utils.floatarray(jid + '.crop_size', 'crop size', def=[32, 32]),
        {
            _id_: jid + '.method',
            name: { en: 'method', cn: self.en },
            type: 'string-enum',
            objs: [
                {
                    name: { en: 'center', cn: self.en },
                    value: 'center',
                },
                {
                    name: { en: 'grid', cn: self.en },
                    value: 'grid',
                    // TODO with trigger
                },
                {
                    name: { en: 'random', cn: self.en },
                    value: 'random',
                },
            ],
            default: self.objs[0].value,
        },
        _Utils.bool(jid + '.allow_outsize_center', 'outsize center', def=true),
    ],
}
