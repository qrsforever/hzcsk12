// @file knn.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-13 14:36

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.int('model.knn.n_neighbors', 'Num Neighbors', def=5, ddd=true),
                _Utils.int('model.knn.leaf_size', 'Leaf Size', def=30, ddd=true),
                {
                    _id_: 'model.knn.weights',
                    name: { en: 'Weights', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'uniform', cn: self.en },
                            value: 'uniform',
                        },
                        {
                            name: { en: 'distance', cn: self.en },
                            value: 'distance',
                        },
                    ],
                    default: 'uniform',
                },
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('model.knn.p', 'Power', def=2, min=1),
                _Utils.int('model.knn.n_jobs', 'Num Jobs', def=1, min=1),
                {
                    _id_: 'model.knn.algorithm',
                    name: { en: 'Algorithm', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: { en: 'auto', cn: self.en },
                            value: 'auto',
                        },
                        {
                            name: { en: 'ball_tree', cn: self.en },
                            value: 'ball_tree',
                        },
                        {
                            name: { en: 'kd_tree', cn: self.en },
                            value: 'kd_tree',
                        },
                        {
                            name: { en: 'brute', cn: self.en },
                            value: 'brute',
                        },
                    ],
                    default: 'auto',
                },
            ],
        },
    ],
}
