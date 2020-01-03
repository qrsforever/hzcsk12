// @file rnn.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-03 15:23

{
    get(jid):: {
        type: 'H',
        objs: (import 'common.libsonnet').get(jid),
    },
}
