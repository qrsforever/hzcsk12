// @file embedding.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 22:42

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid, flag=true):: [
        {
            type: 'H',
            objs: [
                _Utils.int(
                    jid + '.embedding_dim',
                    'embedding dim',
                    def=200,
                    ddd=true,
                    tips='the size of each embedding vector'
                ),
            ] + if flag then [
                _Utils.booltrigger(
                    '_k12.' + jid + '.pretrained_file.bool',
                    'pretrained file',
                    def=false,
                    ddd=true,
                    trigger=[
                        _Utils.stringenum(
                            jid + '.pretrained_file',
                            'glove',
                            def='/pretrained/glove/glove.6B.100d.txt.gz',
                            ddd=true,
                            enums=[
                                { name: { en: '6B.50d', cn: self.en }, value: '/pretrained/glove/glove.6B.50d.txt.gz' },
                                { name: { en: '6B.100d', cn: self.en }, value: '/pretrained/glove/glove.6B.100d.txt.gz' },
                                { name: { en: '6B.200d', cn: self.en }, value: '/pretrained/glove/glove.6B.200d.txt.gz' },
                                { name: { en: '6B.300d', cn: self.en }, value: '/pretrained/glove/glove.6B.300d.txt.gz' },
                                { name: { en: '840B.300d', cn: self.en }, value: '/pretrained/glove/glove.840B.300d.txt.gz' },
                            ],
                            tips='path to a file of word vectors to initialize the embedding matrix'
                        ),
                    ]
                ),
                _Utils.bool(
                    jid + '.trainable',
                    'trainable',
                    def=false,
                    tips='whether or not to optimize the embedding parameters',
                ),
                _Utils.bool(
                    jid + '.sparse',
                    'sparse',
                    def=false,
                    tips='whether or not the Pytorch backend should use a sparse representation of the embedding weight'
                ),
            ] else [],
        },
    ],
}
