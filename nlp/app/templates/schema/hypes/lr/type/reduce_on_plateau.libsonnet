// @file reduce_on_plateau.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 14:04

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): {
        type: 'H',
        objs: [
            {
                _id_: jid + '.mode',
                name: { en: 'Mode', cn: self.en },
                type: 'string-enum',
                objs: [
                    {
                        name: { en: 'min', cn: self.en },
                        value: 'min',
                    },
                    {
                        name: { en: 'max', cn: self.en },
                        value: 'max',
                    },
                ],
                default: self.objs[0].value,
            },
            _Utils.float(jid + '.factor', 'Factor', def=0.1),
            _Utils.int(jid + '.patience', 'Patience', def=10),
        ],
    },
}
