// @file characters.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-22 20:59

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: [
                _Utils.string(jid + '.type', 'type', def='characters', readonly=true),
                _Utils.booltrigger(
                    '_k12.' + jid + '.min_padding_length.bool',
                    'min padding length',
                    def=false,
                    tips='minimum padding length required for the TokenIndexer',
                    trigger=[_Utils.int(jid + '.min_padding_length', 'value', min=0, def=0)]
                ),
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
