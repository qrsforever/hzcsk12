// @file character_encoding.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 23:19

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: (import 'embedding.libsonnet').get(jid + '.embedding', false),
        },
        {
            type: 'H',
            objs: [(import '../encoders/__init__.jsonnet').get(jid + '.encoder')],
        },
    ],
}
