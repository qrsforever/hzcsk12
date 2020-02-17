// @file kmeans.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-14 22:27

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.int('model.kmeans.n_clusters', 'Num Clusters', def=8, ddd=true),
                _Utils.int('model.kmeans.max_iter', 'Max Iter', def=300, min=100, ddd=true),
                _Utils.float('model.kmeans.tol', 'Tolerance', def=1e-4),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('model.kmeans.n_init', 'Num Init', def=10, min=1),
                {
                    _id_: 'model.kmeans.init',
                    name: { en: 'Init Method', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'k-means++', cn: self.en },
                            value: 'k-means++',
                        },
                        {
                            name: { en: 'random', cn: self.en },
                            value: 'random',
                        },
                    ],
                    default: 'k-means++',
                },
                _Utils.bool('model.kmeans.copy_x', 'Copy X', def=true),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('model.kmeans.verbose', 'Verbose', def=0),
                {
                    _id_: 'model.kmeans.algorithm',
                    name: { en: 'Algorithm', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'auto', cn: self.en },
                            value: 'auto',
                        },
                        {
                            name: { en: 'full', cn: self.en },
                            value: 'full',
                        },
                        {
                            name: { en: 'elkan', cn: self.en },
                            value: 'elkan',
                        },
                    ],
                    default: 'auto',
                },
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.booltrigger('_k12.model.kmeans.precompute_distances',
                                   'Precompute Dist',
                                   def=false,
                                   trigger=[_Utils.bool('model.kmeans.precompute_distances', 'Value', def=true, ddd=true)]),
                _Utils.booltrigger('_k12.model.kmeans.n_jobs',
                                   'Jobs',
                                   def=false,
                                   trigger=[_Utils.int('model.kmeans.n_jobs', 'Value', def=1, ddd=true)]),
                _Utils.booltrigger('_k12.model.kmeans.random_state',
                                   'Random State',
                                   def=false,
                                   trigger=[_Utils.int('model.kmeans.random_state', 'Value', def=1, ddd=true)]),
            ],
        },
    ],
}
