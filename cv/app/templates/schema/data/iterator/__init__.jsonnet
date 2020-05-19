// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 14:46

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.int('solver.display_iter', 'Display Iters', min=32, def=60, tips='report metrics every display iters'),
                _Utils.int('solver.test_interval', 'Test Iters', min=64, def=300, tips='use validation dataset every test iters'),
                _Utils.int('solver.save_iters', 'Save Iters', min=128, def=600, tips='save model files every save iters'),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.stringenum('train.batch_size',
                                  'Train Batch Size',
                                  ddd=true,
                                  def=32,
                                  tips='samples count per batch to load on training',
                                  enums=[
                                      { name: { en: '16', cn: self.en }, value: 16 },
                                      { name: { en: '32', cn: self.en }, value: 32 },
                                      { name: { en: '64', cn: self.en }, value: 64 },
                                      { name: { en: '128', cn: self.en }, value: 128 },
                                      { name: { en: '256', cn: self.en }, value: 256 },
                                  ]),
                _Utils.stringenum('val.batch_size',
                                  'Val Batch Size',
                                  ddd=true,
                                  def=32,
                                  tips='samples count per batch to load on validation',
                                  enums=[
                                      { name: { en: '16', cn: self.en }, value: 16 },
                                      { name: { en: '32', cn: self.en }, value: 32 },
                                      { name: { en: '64', cn: self.en }, value: 64 },
                                      { name: { en: '128', cn: self.en }, value: 128 },
                                      { name: { en: '256', cn: self.en }, value: 256 },
                                  ]),
                _Utils.stringenum('test.batch_size',
                                  'Test Batch Size',
                                  ddd=true,
                                  def=32,
                                  tips='samples count per batch to load on evaluating',
                                  enums=[
                                      { name: { en: '16', cn: self.en }, value: 16 },
                                      { name: { en: '32', cn: self.en }, value: 32 },
                                      { name: { en: '64', cn: self.en }, value: 64 },
                                      { name: { en: '128', cn: self.en }, value: 128 },
                                      { name: { en: '256', cn: self.en }, value: 256 },
                                  ]),
            ],
        },
    ],
}
