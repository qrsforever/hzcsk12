// @file bucket.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-26 00:09

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        type: '_ignore_',
        objs: [
            {
                type: 'H',
                objs: [
                    _Utils.string(jid + '.type', 'batch sampler', def='bucket', readonly=true),
                    (import 'basic.libsonnet').batch_size(jid + '.batch_size', 64),
                    _Utils.float(jid + '.padding_noise',
                                 'padding noise',
                                 min=0.001,
                                 max=0.999,
                                 def=0.1,
                                 tips='when sorting by padding length, we add a bit of noise to the lengths'),
                    _Utils.bool(jid + '.drop_last',
                                'drop last',
                                def=false,
                                tips='whether or not drop the last batch if its size would be less the batch_size'),
                    _Utils.booltrigger('_k12.' + jid + '.sorting_keys',
                                       'sorting keys',
                                       def=false,
                                       ddd=true,
                                       tips='value like [str, ...] or [[str, str], ...], to bucket inputs into batches, we want to group the instances by padding length, so that we minimize the amount of padding necessary per batch',
                                       trigger=[
                                           _Utils.stringarray(
                                               jid + '.sorting_keys',
                                               'value',
                                               def=['tokens'],
                                           ),
                                       ]),
                ],
            },
        ],
    },
}
