// @file cluster.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-03-23 16:54

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.booltrigger('_k12.metrics.v_measure_score',
                               'V Measure Score',
                               def=false,
                               tips='the harmonic mean between homogeneity and completeness',
                               trigger=[_Utils.float('metrics.v_measure_score.beta',
                                                     'Beta',
                                                     def=1.0,
                                                     max=1.0,
                                                     min=0.1)]),
            _Utils.bool('_k12.metrics.completeness_score', 'Completeness Score', def=true, tips='metric of a cluster labeling given a ground truth'),
            _Utils.bool('_k12.metrics.homogeneity_score', 'Homogeneity Score', def=true, tips='metric of a cluster labeling given a ground truth'),
        ],
    },
    {
        type: 'H',
        objs: [
            _Utils.booltrigger('_k12.metrics.ami',
                               'AMI',
                               def=false,
                               tips='Adjusted Mutual Information between two clusterings',
                               trigger=[{
                                   _id_: 'metrics.ami.average_method',
                                   name: { en: 'Average Method', cn: self.en },
                                   type: 'string-enum',
                                   objs: [
                                       {
                                           name: { en: 'min', cn: self.en },
                                           value: 'min',
                                       },
                                       {
                                           name: { en: 'max', cn: self.en },
                                           value: 'max',
                                       },
                                       {
                                           name: { en: 'geometric', cn: self.en },
                                           value: 'geometric',
                                       },
                                       {
                                           name: { en: 'arithmetic', cn: self.en },
                                           value: 'arithmetic',
                                       },
                                   ],
                                   default: 'max',
                               }]),
            _Utils.booltrigger('_k12.metrics.fmi',
                               'FMI',
                               def=false,
                               tips='Measure the similarity of two clusterings of a set of points',
                               trigger=[_Utils.bool('metrics.fmi.sparse', 'Sparse', def=false)]),
            _Utils.bool('_k12.metrics.ari', 'ARI', def=true, tips='Rand index adjusted for chance'),
        ],
    },
]
