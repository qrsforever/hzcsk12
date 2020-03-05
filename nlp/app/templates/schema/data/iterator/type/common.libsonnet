// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 23:51

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid)::
        [
            _Utils.stringenum(jid + '.batch_size', 'batch size', def=32, enums=[
                { name: { en: '16', cn: self.en }, value: 16 },
                { name: { en: '32', cn: self.en }, value: 32 },
                { name: { en: '64', cn: self.en }, value: 64 },
                { name: { en: '128', cn: self.en }, value: 128 },
                { name: { en: '256', cn: self.en }, value: 256 },
            ]),
            _Utils.int(jid + '.instances_per_epoch', 'instance per epoch', min=8, def=32),
            _Utils.int(jid + '.max_instances_in_memory', 'max instance', min=8, def=32),
            _Utils.bool(jid + '.cache_instances', 'cache instances', def=false),
            _Utils.bool(jid + '.track_epoch', 'track epoch', def=false),
        ],
}
