// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-11 23:33


local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.string('model.name', 'Algo', def=_Utils.network, readonly=true),
        ],
    },
] + (
    if 'svc' == _Utils.network then
        (import 'types/svc.libsonnet').get()
    else if 'svr' == _Utils.network then
        (import 'types/svr.libsonnet').get()
    else if 'random_forest' == _Utils.network then
        (import 'types/random_forest.libsonnet').get()
    else []
)
