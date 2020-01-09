// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 23:05

local _Utils = import '../utils/helper.libsonnet';

(import 'network/__init__.jsonnet').get()
+
(import 'details/__init__.jsonnet').get()
