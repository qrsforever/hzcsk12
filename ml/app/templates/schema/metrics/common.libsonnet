// @file common.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-05 17:55

local _Utils = import '../utils/helper.libsonnet';

{
    get():: if 'decision_tree' == _Utils.network || 'random_forest' == _Utils.network
    then [
        {
            type: 'H',
            objs: [
                _Utils.bool('_k12.metrics.tree_dot', 'DTree', def=false, tips='set Max Depth < 5'),
                _Utils.booltrigger(
                    '_k12.metrics.dtreeviz',
                    'VTree',
                    def=false,
                    trigger=[
                        {
                            type: 'H',
                            objs: [
                                _Utils.string('metrics.dtreeviz.target_name',
                                              'target name',
                                              def=_Utils.dataset_name,
                                              readonly=true),
                                _Utils.stringenum(
                                    'metrics.dtreeviz.orientation',
                                    'orientation',
                                    def='TD',
                                    enums=[
                                        {
                                            name: { en: 'TD', cn: self.en },
                                            value: 'TD',
                                        },
                                        {
                                            name: { en: 'LR', cn: self.en },
                                            value: 'LR',
                                        },
                                    ]
                                ),
                                _Utils.bool('metrics.dtreeviz.fancy', 'fancy', def=true),
                                _Utils.bool('metrics.dtreeviz.show_node_labels', 'show labels', def=true),
                            ],
                        },
                    ] + (
                        if 'random_forest' == _Utils.network
                        then [
                            _Utils.int('metrics.dtreeviz_estimators_num', 'Estimators Num', def=3, min=1),
                        ]
                        else []
                    ),
                    tips='make visualize look nice'
                ),
            ],
        },
    ]
    else [],
}
