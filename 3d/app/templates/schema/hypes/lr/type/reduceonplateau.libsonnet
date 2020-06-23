// @file reduceonplateau.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-23 22:11

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        name: { en: 'ReduceOnPlateau', cn: self.en },
        type: 'H',
        objs: [
            _Utils.stringenum(jid + '.mode',
                              'Mode',
                              def='min',
                              tips='In min mode: lr will be reduced when the quantity monitored has stopped decreasing',
                              enums=[
                                  { name: { en: 'min', cn: self.en }, value: 'min' },
                                  { name: { en: 'max', cn: self.en }, value: 'max' },
                              ]),
            _Utils.float(jid + '.factor',
                         'Factor',
                         min=0.001,
                         max=0.999,
                         def=0.10,
                         tips='Factor by which the learning rate will be reduced'),

            _Utils.int(jid + '.patience',
                       'Patience',
                       min=5,
                       def=10,
                       tips='Number of epochs with no improvement after which learning rate will be reduced'),

            _Utils.float(jid + '.eps',
                         'EPS',
                         min=0.001,
                         max=0.999,
                         def=1e-8,
                         tips='Minimal decay applied to lr'),
        ],
    },
}
