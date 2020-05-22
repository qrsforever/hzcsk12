// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-27 15:04

local _Utils = import '../../utils/helper.libsonnet';

local _READERS = {
    sst: {
        get(jid):: (import 'sst.libsonnet').get(jid),
    },  // dataset_name: sst
    conll2003: {
        get(jid):: (import 'conll2003.libsonnet').get(jid),
    },
};

{
    get():: {
        name: { en: 'Phase', cn: self.en },
        type: 'navigation',
        objs: [
            {
                local jid = 'dataset_reader',
                name: { en: 'Train', cn: self.en },
                type: '_ignore_',
                objs: _READERS[_Utils.dataset_name].get(jid) +
                      [
                          _Utils.booltrigger(
                              '_k12.' + jid + '.single_id.bool',
                              'single id',
                              def=false,
                              tips='represents tokens as single integers',
                              trigger=(import 'indexers/single_id.libsonnet').get(jid + '.token_indexers.tokens'),
                          ),
                          _Utils.booltrigger(
                              '_k12.' + jid + '.token_characters.bool',
                              'characters',
                              def=false,
                              tips='represents tokens as lists of character indices',
                              trigger=(import 'indexers/characters.libsonnet').get(jid + '.token_indexers.token_characters'),
                          ),
                          _Utils.string('train_data_path',
                                        'Dataset Path',
                                        width=500,
                                        readonly=true),
                      ],
            },
            {
                local jid = 'validation_dataset_reader',
                name: { en: 'Validation', cn: self.en },
                type: '_ignore_',
                objs: [
                    {
                        _id_: '_k12.' + jid + '.bool',
                        name: { en: 'Enable', cn: self.en },
                        type: 'bool-trigger',
                        objs: [
                            {
                                value: true,
                                trigger: {
                                    type: '_ignore_',
                                    objs: _READERS[_Utils.dataset_name].get(jid) +
                                          [
                                              _Utils.string('validation_data_path',
                                                            'Dataset Path',
                                                            width=500,
                                                            readonly=true),
                                              _Utils.booltrigger(
                                                  '_k12.' + jid + '.single_id.bool',
                                                  'single id',
                                                  def=false,
                                                  tips='represents tokens as single integers',
                                                  trigger=(import 'indexers/single_id.libsonnet').get(jid + '.token_indexers.tokens'),
                                              ),
                                              _Utils.booltrigger(
                                                  '_k12.' + jid + '.token_characters.bool',
                                                  'characters',
                                                  def=false,
                                                  tips='represents tokens as lists of character indices',
                                                  trigger=(import 'indexers/characters.libsonnet').get(jid + '.token_indexers.token_characters'),
                                              ),
                                          ],
                                },
                            },
                            {
                                value: false,
                                trigger: {},
                            },
                        ],
                        default: _Utils.get_default_value(self._id_, false),
                    },
                ],
            },
            {
                name: { en: 'Evaluate', cn: self.en },
                type: '_ignore_',
                objs: [
                    _Utils.string('test_data_path',
                                  'Dataset Path',
                                  def='',
                                  ddd=true,
                                  width=500,
                                  readonly=true),
                ],
            },
        ],
    },
}
