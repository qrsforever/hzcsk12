// @file test.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-27 17:36

local _BASIC = import 'utils/basic_type.libsonnet';

{
    a: _BASIC.bool(1, 'bool-1'),
}
