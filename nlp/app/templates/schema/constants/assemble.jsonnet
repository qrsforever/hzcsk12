// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 23:22

local common = import 'common.libsonnet';

if common.dataset_name == 'sst' then
    import 'sst.jsonnet'

else if common.dataset_name == 'xxx' then
    import 'xxx.jsonnet'
