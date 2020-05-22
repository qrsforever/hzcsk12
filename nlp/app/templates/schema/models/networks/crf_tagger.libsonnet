// @file crf_tagger.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-22 21:20

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: [
                _Utils.stringenum(
                    jid + '.label_encoding',
                    'label encoding',
                    def='BIOUL',
                    ddd=true,
                    enums=[
                        { name: { en: 'BIO', cn: self.en }, value: 'BIO' },
                        { name: { en: 'BIOUL', cn: self.en }, value: 'BIOUL' },
                        { name: { en: 'IOB1', cn: self.en }, value: 'IOB1' },
                        { name: { en: 'BMES', cn: self.en }, value: 'BMES' },
                    ],
                    tips='label encoding to use when calculating span f1 and constraining the CRF at decoding time'
                ),
                _Utils.bool(
                    jid + '.constrain_crf_decoding',
                    'CRF constrained',
                    def=false,
                    ddd=true,
                    tips='if True the CRF is constrained at decoding time to produce valid sequences of tags'
                ),
                _Utils.bool(
                    jid + '.calculate_span_f1',
                    'calculate F1',
                    def=false,
                    ddd=true,
                    tips='calculate span-level F1 metrics during training'
                ),
                _Utils.bool(
                    jid + '.include_start_end_transitions',
                    'transition',
                    def=false,
                    ddd=true,
                    tips='whether to include start and end transition parameters in the CRF'
                ),
                _Utils.int(
                    jid + '.top_k',
                    'top k',
                    def=1,
                    min=1,
                    ddd=true,
                    tips='the number of parses to return from the crf'
                ),
                _Utils.booltrigger(
                    '_k12.' + jid + '.dropout.bool',
                    'dropout layer',
                    def=false,
                    ddd=true,
                    trigger=[_Utils.float(
                        jid + '.dropout',
                        'value',
                        def=0.5,
                        min=0.001,
                        max=1,
                        tips='nn.dropout',
                    )]
                ),
            ],
        },
        {
            type: 'accordion',
            objs: [
                {
                    name: { en: 'Embedder', cn: self.en },
                    type: '_ignore_',
                    objs: [
                        {
                            local jid1 = jid + '.text_field_embedder.token_embedders',
                            name: { en: 'Embeders', cn: self.en },
                            type: 'navigation',
                            objs: [
                                {
                                    name: { en: 'embedding', cn: self.en },
                                    type: '_ignore_',
                                    objs: [
                                              _Utils.string(
                                                  jid1 + '.tokens.type',
                                                  'type',
                                                  def='embedding',
                                                  readonly=true
                                              ),
                                          ] +
                                          (import '../embedders/embedding.libsonnet').get(jid1 + '.tokens'),
                                },
                                {
                                    name: { en: 'character', cn: self.en },
                                    type: '_ignore_',
                                    objs: [
                                        _Utils.booltrigger(
                                            '_k12.' + jid1 + '.bool',
                                            'Enable',
                                            def=true,
                                            tips='character text field embeders',
                                            trigger=[
                                                _Utils.string(
                                                    jid1 + '.token_characters.type',
                                                    'type',
                                                    def='character_encoding',
                                                    readonly=true
                                                ),
                                            ] + (import '../embedders/character_encoding.libsonnet').get(jid1 + '.token_characters'),
                                        ),
                                    ],
                                },
                                {
                                    name: { en: 'elmo', cn: self.en },
                                    type: '_ignore_',
                                    objs: [
                                        _Utils.booltrigger(
                                            '_k12.' + jid1 + '.bool',
                                            'Enable',
                                            def=false,
                                            readonly=true,
                                            tips='elmo text field embeders',
                                            trigger=(import '../embedders/elmo_token_embedder.libsonnet').get(jid1 + '.elmo'),
                                        ),
                                    ],
                                },
                            ],
                        },  // embedders
                    ],
                },
                {
                    name: { en: 'Encoder', cn: self.en },
                    type: '_ignore_',
                    objs: [
                        {
                            local jid2 = jid,
                            name: { en: 'Encoders', cn: self.en },
                            type: 'navigation',
                            objs: [
                                {
                                    name: { en: 'encoder', cn: self.en },
                                    type: '_ignore_',
                                    objs: [
                                        (import '../encoders/__init__.jsonnet').get(jid2 + '.encoder'),
                                    ],
                                },
                            ],
                        },  // encoders
                    ],
                },
            ],
        },
    ],
}
