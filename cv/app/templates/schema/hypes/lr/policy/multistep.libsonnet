// @file multistep.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 22:20

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid):: {
        name: { en: 'MultiStep Parameters', cn: self.en },
        type: 'H',
        objs: [
            _Utils.float(jid + '.gamma',
                         'Gamma',
                         min=0.001,
                         max=0.999,
                         def=0.10,
                         tips='multiplicative factor of learning rate decay'),
            _Utils.intarray(
                jid + '.stepvalue',
                'Milestones',
                def=[90, 120],
                tips='value like [int, int, ...], the element is one of of iters indices and all elements is increasing in turn'
            ),
        ],
    },
}
