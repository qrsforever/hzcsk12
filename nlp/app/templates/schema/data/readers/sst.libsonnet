// @file sst.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 20:33

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid, navi):: [
        // if navi == 'train' then
        //     _Utils.string('train_data_path',
        //                   'Data Path',
        //                   def=_Utils.dataset_path + '/sst/train.txt',
        //                   readonly=true) else
        //     _Utils.string('validation_data_path',
        //                   'Data Path',
        //                   def=_Utils.dataset_path + '/sst/dev.txt',
        //                   readonly=true),
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
                        type: '_ignore_',
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
