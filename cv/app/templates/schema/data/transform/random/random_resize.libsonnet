// @file random_resize.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 19:38

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.radio', 'radio', def=0.5),
        _Utils.floatarray(jid + '.scale_range', 'scale range', def=[0.75, 1.25]),
        _Utils.floatarray(jid + '.aspect_range', 'aspect range', def=[0.9, 1.1]),
        {
            _id_: jid + '.method',
            name: { en: 'method', cn: self.en },
            type: 'string-enum',
            objs: [
                {
                    name: { en: 'random', cn: self.en },
                    value: 'random',
                },
                {
                    name: { en: 'focus', cn: self.en },
                    value: 'focus',
                },
                {
                    name: { en: 'bound', cn: self.en },
                    value: 'bound',
                },
            ],
            default: self.objs[0].value,
        },
    ],
}
