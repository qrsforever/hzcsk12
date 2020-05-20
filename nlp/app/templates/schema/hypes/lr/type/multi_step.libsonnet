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
            _Utils.intarray(
                jid + '.milestones',
                'Milestones',
                def=[90, 120],
                tips='value like [int, int, ...], the element is one of of epoch indices and all elements is increasing in turn'
            ),
            _Utils.float(jid + '.gamma',
                         'Gamma',
                         min=0.001,
                         max=0.999,
                         def=0.10,
                         tips='multiplicative factor of learning rate decay'),
        ],
    },
}
