// @file basic.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 00:02

{
    get(jid, dataset_name): {
        type: 'H',
        objs: (import 'common.libsonnet').get(jid),
    },
}
