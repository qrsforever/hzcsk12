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
                    // _Utils.stringarray(jid + '.sorting_keys',
                    //                    'sorting keys',
                    //                    def=_KEYS[_Utils.dataset_name],
                    //                    width=500,
                    //                    readonly=true),
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
                ],
            },
        ],
    },
}
