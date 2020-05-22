// @file single_id.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-27 16:58

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: [
                _Utils.string(jid + '.type', 'type', def='single_id', readonly=true),
                _Utils.bool(jid + '.lowercase_tokens',
                            'lowercase',
                            def=false,
                            tips='token convert to lowercase before getting an index for the token from the vocabulary'),
                _Utils.booltrigger(
                    '_k12.' + jid + '.token_min_padding_length.bool',
                    'token min padding',
                    def=false,
                    tips='minimum padding length required for the TokenIndexer',
                    trigger=[_Utils.int(jid + '.token_min_padding_length', 'value', min=0, def=0)]
                ),
            ],
        },
    ],
}
