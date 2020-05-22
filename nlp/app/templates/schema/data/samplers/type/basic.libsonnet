// @file basic.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-22 17:54


local _Utils = import '../../../utils/helper.libsonnet';

{
    batch_size(jid, def=32)::
        _Utils.stringenum(jid,
                          'batch size',
                          def=def,
                          enums=[
                              { name: { en: '16', cn: self.en }, value: 16 },
                              { name: { en: '32', cn: self.en }, value: 32 },
                              { name: { en: '64', cn: self.en }, value: 64 },
                              { name: { en: '128', cn: self.en }, value: 128 },
                              { name: { en: '256', cn: self.en }, value: 256 },
                          ],
                          tips='the size of each batch of instances yeilded'),
    get(jid, flag):: {
        type: '_ignore_',
        objs: [
            {
                type: 'H',
                objs: [
                    $.batch_size(jid + '.batch_size', 64),
                    _Utils.bool(jid + '.drop_last',
                                'drop last',
                                def=false,
                                tips='whether or not drop the last batch if its size would be less the batch_size'),
                    _Utils.bool(
                        jid + '.shuffle',
                        'shuffle',
                        readonly=flag,
                        def=false,
                        tips='reshuffle the data at every epoch',
                    ),
                ],
            },
        ],
    },
}
