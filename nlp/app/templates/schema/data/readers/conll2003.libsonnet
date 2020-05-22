// @file conll2003.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-22 19:03

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: [
                _Utils.string(jid + '.type', 'type', def='conll2003', readonly=true),
                _Utils.stringenum(jid + '.tag_label',
                                  'tag label',
                                  def='ner',
                                  enums=[
                                      { name: { en: 'ner', cn: self.en }, value: 'ner' },
                                      { name: { en: 'pos', cn: self.en }, value: 'pos' },
                                      { name: { en: 'chunk', cn: self.en }, value: 'chunk' },
                                  ],
                                  tips='loaded into the instance field tag'),
                _Utils.stringenum(jid + '.coding_scheme',
                                  'coding scheme',
                                  def='BIOUL',
                                  enums=[
                                      { name: { en: 'IOB1', cn: self.en }, value: 'IOB1' },
                                      { name: { en: 'BIOUL', cn: self.en }, value: 'BIOUL' },
                                  ],
                                  tips='specifies the coding scheme for ner_labels and chunk_labels'),
                _Utils.string(jid + '.label_namespace',
                              'label namespace',
                              def='labels',
                              readonly=true,
                              tips='specify the namespace for the chosen tag label'),
            ],
        },
    ],
}
