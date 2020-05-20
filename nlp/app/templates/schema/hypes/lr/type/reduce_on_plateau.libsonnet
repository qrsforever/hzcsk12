// @file reduce_on_plateau.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 14:04

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'H',
        objs: [
            {
                _id_: jid + '.mode',
                name: { en: 'Mode', cn: self.en },
                type: 'string-enum',
                tips: 'in min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. ',
                objs: [
                    {
                        name: { en: 'min', cn: self.en },
                        value: 'min',
                    },
                    {
                        name: { en: 'max', cn: self.en },
                        value: 'max',
                    },
                ],
                default: self.objs[0].value,
            },
            _Utils.float(jid + '.factor',
                         'Factor',
                         min=0.001,
                         max=0.999,
                         def=0.1,
                         tips='factor by which the learning rate will be reduced. new_lr = lr * factor.'),
            _Utils.int(jid + '.patience',
                       'Patience',
                       min=1,
                       def=10,
                       tips='number of epochs with no improvement after which learning rate will be reduced'),
        ],
    },
}
