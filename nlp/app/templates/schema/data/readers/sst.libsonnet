// @file sst.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 20:33

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid, navi):: [
        _Utils.string(jid + '.granularity', 'granularity', def='5-class', tips='2-class, 3-class, 5-class'),
        _Utils.bool(jid + '.lazy', 'lazy', def=false),
        _Utils.bool(jid + '.use_subtrees', 'use subtrees', def=false),
        {
            _id_: '_k12.token_indexers.single_id',
            name: { en: 'single_id', cn: self.en },
            type: 'bool-trigger',
            objs: [
                {
                    value: true,
                    trigger: {
                        type: '_ignore_',
                        objs: (import 'indexers/single_id.libsonnet').get(jid + '.token_indexers.tokens'),
                    },
                },
                {
                    value: false,
                    trigger: {},
                },
            ],
            default: false,
        },
        if navi == 'train' then
            _Utils.string('train_data_path',
                          'Dataset Path',
                          def='/datasets/sst/train.txt',
                          ddd=true,
                          width=500,
                          readonly=true) else
            _Utils.string('validation_data_path',
                          'Dataset Path',
                          def='/datasets/sst/dev.txt',
                          ddd=true,
                          width=500,
                          readonly=true),
    ],
}
