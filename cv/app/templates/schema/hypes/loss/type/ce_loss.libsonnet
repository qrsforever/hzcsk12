// @file ce_loss.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 21:55

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        name: { en: 'CE Parameters', cn: self.en },
        type: 'H',
        objs: [
            _Utils.float('loss.loss_weights.ce_loss.ce_loss',
                         'Weight',
                         min=0.0,
                         max=1.0,
                         def=1.0,
                         tips='a manual rescaling weight given to each loss class.'),
            {
                _id_: jid + '.reduction',
                type: 'string-enum',
                name: { en: 'Reduction', cn: self.en },
                tips: 'specifies the reduction to apply to the output',
                objs: [
                    { name: { en: 'mean', cn: self.en }, value: 'mean' },
                    { name: { en: 'sum', cn: self.en }, value: 'sum' },
                    { name: { en: 'none', cn: self.en }, value: 'none' },
                ],
                default: 'mean',
            },
            _Utils.int(jid + '.ignore_index',
                       'Ignore Index',
                       def=-1,
                       tips='specifies a target value that is ignored and does not contribute to the input gradient.'),
        ],
    },
}
