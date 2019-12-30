// @file sst.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 20:33

// train_data_path: '/data/datasets/nlp/sst/train.txt',
// validation_data_path: '/data/datasets/nlp/sst/dev.txt',
// test_data_path: '/data/datasets/nlp/sst/test.txt',

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid): [
        {
            type: 'H',
            objs: [
                _Utils.bool(jid + '.layz', 'lazy', def=true),
                _Utils.bool(jid + '.use_subtrees', 'use subtrees', def=false),
                _Utils.string(jid + '.granularity', 'granularity', def='5-class'),
            ],
        },
        {
            local tokenid = jid + 'token_indexers.tokens',
            _id_: '_k12.token_indexers.single_id',
            name: { en: 'single_id', cn: self.en },
            type: 'bool-trigger',
            objs: [
                {
                    value: true,
                    trigger: {
                        objs: (import 'indexers/single_id.libsonnet').get(tokenid),
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
