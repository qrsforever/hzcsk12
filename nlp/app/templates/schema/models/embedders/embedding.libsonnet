// @file embedding.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 22:42

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: [
                _Utils.string(
                    jid + '.type',
                    'Type',
                    def='embedding',
                    readonly=true
                ),
                _Utils.int(
                    jid + '.embedding_dim',
                    'Embedding Dim',
                    def=200,
                ),
                _Utils.bool(
                    jid + '.trainable',
                    'Trainable',
                    def=false,
                ),
            ],
        },
        {
            _id_: '_k12.' + jid + '.pretrained_file.bool',
            name: { en: 'Use Pretrained File', cn: self.en },
            type: 'bool-trigger',
            objs: [
                {
                    value: true,
                    trigger: {
                        type: '_ignore_',
                        objs: [
                            {
                                _id_: jid + '.pretrained_file',
                                name: { en: 'Glove', cn: self.en },
                                type: 'string-enum',
                                objs: [
                                    {
                                        name: { en: '6B.50d', cn: self.en },
                                        value: _Utils.dataset_path + '/glove/glove.6B.50d.txt.gz',
                                    },
                                    {
                                        name: { en: '6B.100d', cn: self.en },
                                        value: _Utils.dataset_path + '/glove/glove.6B.100d.txt.gz',
                                    },
                                    {
                                        name: { en: '6B.200d', cn: self.en },
                                        value: _Utils.dataset_path + '/glove/glove.6B.200d.txt.gz',
                                    },
                                    {
                                        name: { en: '6B.300d', cn: self.en },
                                        value: _Utils.dataset_path + '/glove/glove.6B.300d.txt.gz',
                                    },
                                    {
                                        name: { en: '840B.300d', cn: self.en },
                                        value: _Utils.dataset_path + '/glove/glove.840B.300d.txt.gz',
                                    },
                                ],
                                default: self.objs[2].value,
                                tips: 'pretrained file',
                            },
                        ],
                    },
                },
                {
                    value: false,
                    trigger: {},
                },
            ],
            default: false,
        },
    ],
}
