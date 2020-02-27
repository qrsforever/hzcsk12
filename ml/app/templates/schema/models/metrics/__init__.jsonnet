// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-27 16:56

local _Utils = import '../../utils/helper.libsonnet';

{
    get()::
        if 'classifier' == _Utils.task
        then
            import 'classifier.libsonnet'
        else if 'regressor' == _Utils.task
        then
            import 'regressor.libsonnet'
        else [],
}
