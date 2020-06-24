// @file resize.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-24 17:40

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        {
            type: 'H',
            objs: [
                _Utils.intarray(jid + '.args.size', 'Size', def=[28, 28], tips='Desired output size'),
                _Utils.stringenum(jid + '.args.interpolation',
                                  'Interpolation',
                                  def=2,
                                  enums=[
                                      { name: { en: 'NEAREST', cn: self.en }, value: 0 },
                                      { name: { en: 'LANCZOS', cn: self.en }, value: 1 },
                                      { name: { en: 'BILINEAR', cn: self.en }, value: 2 },
                                      { name: { en: 'BICUBIC', cn: self.en }, value: 3 },
                                      { name: { en: 'BOX', cn: self.en }, value: 4 },
                                      { name: { en: 'HAMMING', cn: self.en }, value: 5 },
                                  ]),
            ],
        },
        _Utils.checkboxphase(jid),
    ],
}
