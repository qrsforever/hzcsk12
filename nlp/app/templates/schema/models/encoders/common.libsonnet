// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-03 15:25

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: [
                _Utils.int(jid + '.input_size', 'input size', def=200, ddd=true),
                _Utils.int(jid + '.hidden_size', 'hidden size', def=512, ddd=true),
                _Utils.int(jid + '.num_layers', 'num layers', def=2),
                _Utils.int(jid + '.dropout', 'dropout', def=0),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.bool(jid + '.bias', 'bias', def=true),
                _Utils.bool(jid + '.bidirectional', 'bidirectional', def=false),
            ],
        },
    ],
}
