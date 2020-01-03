// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-03 16:41

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        _id_: jid + '.type',
        name: { en: 'Type', cn: self.en },
        type: 'string-enum-trigger',
        objs: [
            {
                name: { en: 'lstm', cn: self.en },
                value: 'lstm',
                trigger: (import 'lstm.libsonnet').get(jid),
            },
            {
                name: { en: 'gru', cn: self.en },
                value: 'gru',
                trigger: (import 'lstm.libsonnet').get(jid),
            },
            {
                name: { en: 'rnn', cn: self.en },
                value: 'rnn',
                trigger: (import 'rnn.libsonnet').get(jid),
            },
        ],
        default: _Utils.get_default_value(self._id_, self.objs[0].value),
    },
}
