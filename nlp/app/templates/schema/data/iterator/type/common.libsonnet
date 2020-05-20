// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 23:51

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid)::
        [
            _Utils.stringenum(jid + '.batch_size',
                              'batch size',
                              def=32,
                              enums=[
                                  { name: { en: '16', cn: self.en }, value: 16 },
                                  { name: { en: '32', cn: self.en }, value: 32 },
                                  { name: { en: '64', cn: self.en }, value: 64 },
                                  { name: { en: '128', cn: self.en }, value: 128 },
                                  { name: { en: '256', cn: self.en }, value: 256 },
                              ],
                              tips='the size of each batch of instances yeilded'),
            _Utils.booltrigger(
                '_k12.' + jid + '.instances_per_epoch.enable',
                'instances/epoch',
                def=false,
                trigger=[
                    _Utils.int(jid + '.instances_per_epoch',
                               'epoch instances',
                               min=16,
                               def=64,
                               tips='each epoch will consist of precisely this many instances'),
                ],
            ),
            _Utils.booltrigger(
                '_k12.' + jid + '.max_instances_in_memory.enable',
                'instances/memory',
                def=false,
                trigger=[
                    _Utils.int(jid + '.max_instances_in_memory',
                               'max instances',
                               min=64,
                               def=128,
                               tips='load this many instances at a time into an in-memory list'),
                ]
            ),
            _Utils.bool(jid + '.cache_instances',
                        'cache instances',
                        def=false,
                        tips='whether or not to cache the tensorized instances in memory'),
            _Utils.bool(jid + '.track_epoch',
                        'track epoch',
                        def=false,
                        tips='each instance containing the epoch number'),
        ],
}
