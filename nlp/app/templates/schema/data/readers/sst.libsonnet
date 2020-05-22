// @file sst.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 20:33

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: [
                _Utils.string(jid + '.type', 'type', def='sst_tokens', readonly=true),
                _Utils.stringenum(jid + '.granularity',
                                  'granularity',
                                  def='5-class',
                                  enums=[
                                      { name: { en: '2-class', cn: self.en }, value: '2-class' },
                                      { name: { en: '3-class', cn: self.en }, value: '3-class' },
                                      { name: { en: '5-class', cn: self.en }, value: '5-class' },
                                  ],
                                  tips='indicate the number of sentiment labels to use'),
                _Utils.bool(jid + '.lazy',
                            'lazy',
                            def=false,
                            tips='whether or not instances can be read lazily'),
                _Utils.bool(jid + '.use_subtrees',
                            'use subtrees',
                            def=false,
                            tips='whether or not to use sentiment-tagged subtrees'),
            ],
        },
    ],
}
