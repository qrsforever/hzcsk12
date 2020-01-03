// @file multi_step.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 14:07

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'H',
        objs: [
            _Utils.intarray(jid + '.milestones',
                            'Milestones',
                            def=[30, 80],
                            tips='List of epoch indices'),
            _Utils.float(jid + '.gamma',
                         'Gamma',
                         def=0.1,
                         tips='Multiplicative factor of learning rate decay'),
        ],
    },
}
