// @file bucket.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 00:09

local _BASIC = import '../../../utils/basic_type.libsonnet';

local _KEYS = {
    sst: [['tokens', 'num_tokens']],
};

{
    get(jid, dataset_name): {
        objs: [
            {
                type: 'H',
                objs: (import 'common.libsonnet').get(jid),
            },
            {
                type: 'H',
                objs: [
                    _BASIC.stringarray(jid + '.sorting_keys',
                                       'sorting keys',
                                       def=_KEYS[dataset_name],
                                       width=600,
                                       readonly=true),
                    _BASIC.float(jid + '.padding_noise', 'padding noise', def=0.1),
                    _BASIC.bool(jid + '.biggest_batch_first', 'biggest batch first', def=false),
                    _BASIC.bool(jid + '.skip_smaller_batches', 'skip smaller batches', def=false),
                ],
            },
        ],
    },
}
