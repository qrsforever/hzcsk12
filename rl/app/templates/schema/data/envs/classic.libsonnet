// @file classic.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-19 21:45

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid, label): [
        {
            name: { en: label, cn: self.en },
            type: 'object',
            objs: [
                {
                    type: 'H',
                    objs: [
                        _Utils.string(jid + '.id', 'Game', def=_Utils.dataset_name, readonly=true),
                    ],
                },
            ],
        },
    ],
}
