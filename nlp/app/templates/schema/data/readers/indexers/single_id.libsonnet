// @file single_id.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-27 16:58

local _BASIC = import '../../../utils/basic_type.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: [
                _BASIC.string(jid + '.type', 'type', def='single_id', readonly=true),
                _BASIC.bool(jid + '.lowercase_tokens', 'lowercase', def=false),
                _BASIC.int(jid + '.token_min_padding_length', 'min padding length', def=0),
            ],
        },
    ],
}
