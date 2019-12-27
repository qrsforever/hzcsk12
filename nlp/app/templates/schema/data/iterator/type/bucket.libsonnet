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
    get(dataset): {
        objs: [
            {
                type: 'H',
                objs: import 'common.libsonnet',
            },
            {
                type: 'H',
                objs: [
                    _BASIC.stringarray('iterator.sorting_keys',
                                       'sorting keys',
                                       def=_KEYS[dataset],
                                       width=600,
                                       readonly=true),
                    _BASIC.float('iterator.padding_noise', 'padding noise', def=0.1),
                    _BASIC.bool('iterator.biggest_batch_first', 'biggest batch first', def=false),
                    _BASIC.bool('iterator.skip_smaller_batches', 'skip smaller batches', def=false),
                ],
            },
        ],
    },
}
