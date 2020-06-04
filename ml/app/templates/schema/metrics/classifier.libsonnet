// @file classifier.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-27 17:15

local _Utils = import '../utils/helper.libsonnet';

local _Average(method) = {
    _id_: 'metrics.' + method + '.average',
    name: { en: 'Average', cn: self.en },
    type: 'string-enum',
    objs: if 2 == _Utils.get_default_value('data.num_classes', -1)
    then
        [
            {
                name: { en: 'binary', cn: self.en },
                value: 'binary',
            },
        ]
    else
        [
            {
                name: { en: 'micro', cn: self.en },
                value: 'micro',
            },
            {
                name: { en: 'macro', cn: self.en },
                value: 'macro',
            },
            {
                name: { en: 'weighted', cn: self.en },
                value: 'weighted',
            },
        ],
    default: self.objs[0].value,
};

[
    {
        type: 'H',
        objs: [
            _Utils.booltrigger('_k12.metrics.f1',
                               'F1',
                               def=false,
                               trigger=[_Average('f1')]),
            _Utils.booltrigger('_k12.metrics.precision',
                               'Precision',
                               def=false,
                               trigger=[_Average('precision')]),
            _Utils.bool('_k12.metrics.kappa', 'Kappa', def=false, tips="Cohen's kappa: a statistic that measures inter-annotator agreement"),
        ],
    },
    {
        type: 'H',
        objs: [
            _Utils.booltrigger('_k12.metrics.recall',
                               'Recall',
                               def=false,
                               trigger=[_Average('recall')]),
            _Utils.booltrigger('_k12.metrics.jaccard',
                               'Jaccard',
                               def=false,
                               trigger=[_Average('jaccard')]),
            _Utils.bool('_k12.metrics.mcc', 'MCC', def=false, tips='Matthews correlation coefficient'),
        ],
    },
    {
        type: 'H',
        objs: [
            _Utils.booltrigger(
                '_k12.metrics.accuracy',
                'ACC',
                def=true,
                trigger=[_Utils.bool('metrics.accuracy.normalize', 'Normalize', def=true)],
            ),
            _Utils.booltrigger('_k12.metrics.fbeta',
                               'FBeta',
                               def=false,
                               trigger=[
                                   {
                                       type: 'H',
                                       objs: [
                                           _Average('fbeta'),
                                           _Utils.float('metrics.fbeta.beta', 'Beta', def=0.5),
                                       ],
                                   },
                               ]),
            _Utils.bool('_k12.metrics.confusion_matrix', 'Confusion Matrix', def=false),
        ],
    },
] +
if 'decision_tree' == _Utils.network
then
    [
        {
            type: 'H',
            objs: [
                _Utils.booltrigger('_k12.metrics.dtreeviz',
                                   'Nice Tree',
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
                                   ],
                                   tips='make visualize look nice'),
                _Utils.bool('_k12.metrics.tree_dot', 'Display Tree', def=false, tips='set Max Depth < 5'),
            ],
        },
    ]
else []