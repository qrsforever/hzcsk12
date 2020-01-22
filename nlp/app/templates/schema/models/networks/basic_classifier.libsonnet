// @file basic_classifier.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 21:22

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
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
                                objs: (import '../embedders/embedding.libsonnet').get(jid1 + '.tokens'),
                            },
                            {
                                name: { en: 'character', cn: self.en },
                                type: '_ignore_',
                                objs: [
                                    {
                                        _id_: '_k12.' + jid1 + '.bool',
                                        name: { en: 'Enable', cn: self.en },
                                        type: 'bool-trigger',
                                        objs: [
                                            {
                                                value: true,
                                                trigger: {
                                                    type: '_ignore_',
                                                    objs: (import '../embedders/character_encoding.libsonnet').get(jid1 + '.token_characters'),
                                                },
                                            },
                                            {
                                                value: false,
                                                trigger: {},
                                            },
                                        ],
                                        default: false,
                                        readonly: true,
                                    },
                                ],
                            },
                            {
                                name: { en: 'elmo', cn: self.en },
                                type: '_ignore_',
                                objs: [
                                    {
                                        _id_: '_k12.' + jid1 + '.bool',
                                        name: { en: 'Enable', cn: self.en },
                                        type: 'bool-trigger',
                                        objs: [
                                            {
                                                value: true,
                                                trigger: {
                                                    type: '_ignore_',
                                                    objs: (import '../embedders/elmo_token_embedder.libsonnet').get(jid1 + '.elmo'),
                                                },
                                            },
                                            {
                                                value: false,
                                                trigger: {},
                                            },
                                        ],
                                        default: false,
                                        readonly: true,
                                    },
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
                                name: { en: 'seq2vec', cn: self.en },
                                type: '_ignore_',
                                objs: [
                                    (import '../encoders/__init__.jsonnet').get(jid2 + '.seq2vec_encoder'),
                                ],
                            },
                            {
                                name: { en: 'encoder', cn: self.en },
                                type: '_ignore_',
                                objs: [
                                    {
                                        _id_: '_k12.' + jid2 + '.encoder.bool',
                                        name: { en: 'Enable', cn: self.en },
                                        type: 'bool-trigger',
                                        objs: [
                                            {
                                                value: true,
                                                trigger: {
                                                    type: '_ignore_',
                                                    objs: [
                                                        (import '../encoders/__init__.jsonnet').get(jid2 + '.encoder'),
                                                    ],
                                                },
                                            },
                                            {
                                                value: false,
                                                trigger: {},
                                            },
                                        ],
                                        readonly: true,
                                        default: false,
                                    },
                                ],
                            },
                        ],
                    },  // encoders
                ],
            },
        ],
    },
}
