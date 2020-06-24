// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-22 19:21

local _Utils = import '../../utils/helper.libsonnet';

local _base_loss(jid) = {
    type: 'H',
    objs: [
        _Utils.stringenum(jid + '.reduction',
                          'Reduction',
                          def='mean',
                          tips='Specifies the reduction to apply to the output',
                          enums=[
                              { name: { en: 'none', cn: self.en }, value: 'none' },
                              { name: { en: 'mean', cn: self.en }, value: 'mean' },
                              { name: { en: 'sum', cn: self.en }, value: 'sum' },
                          ]),
    ],
};

{
    get(jid):: {
        type: 'V',
        objs: [
            {
                _id_: jid + '.type',
                name: { en: 'Loss Type', cn: self.en },
                type: 'string-enum-trigger',
                objs: [
                    {
                        name: { en: 'MSE', cn: self.en },
                        value: 'mse',
                        trigger: _base_loss(jid + '.args'),
                    },
                    {
                        name: { en: 'CE', cn: self.en },
                        value: 'ce',
                        trigger: _base_loss(jid + '.args'),
                    },
                    {
                        name: { en: 'MaskedMSE', cn: self.en },
                        value: 'maskedmse',
                        trigger: {},
                    },
                    {
                        name: { en: 'BerHu', cn: self.en },
                        value: 'berhu',
                        trigger: {},
                    },
                ],
                default: 'maskedmse',
            },
        ],
    },
}
