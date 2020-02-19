// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-18 00:47

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: if _Utils.network == 'dqn' then
        [import 'dqn.libsonnet']
    else if _Utils.network == 'pg' then
        [import 'pg.libsonnet']
    else [],
}
