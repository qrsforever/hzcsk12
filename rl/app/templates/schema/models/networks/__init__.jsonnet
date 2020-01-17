// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 00:47

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                {
                    _id_: '_k12.models.model_name',
                    name: { en: 'Network', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: _Utils.network_name,
                            value: _Utils.network,
                        },
                    ],
                    default: self.objs[0].value,
                    readonly: true,
                },
                _Utils.bool('model.dueling', 'Dueling', def=false),
                _Utils.bool('algo.double_dqn', 'Double', def=false),
            ],
        },
    ] + if _Utils.network == 'dqn'
    then
        [import 'dqn.libsonnet']
    else [],
}
