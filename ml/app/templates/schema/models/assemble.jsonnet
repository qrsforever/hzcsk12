// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-11 23:33


local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.string('model.name', 'Network', def=_Utils.network, readonly=true),
        ],
    },
] + (
    if 'svc' == _Utils.network then
        (import 'types/svc.libsonnet').get()
    else if 'svr' == _Utils.network then
        (import 'types/svr.libsonnet').get()
    else if 'knn' == _Utils.network then
        (import 'types/knn.libsonnet').get()
    else if 'kmeans' == _Utils.network then
        (import 'types/kmeans.libsonnet').get()
    else if 'gaussian_nb' == _Utils.network then
        (import 'types/gaussian_nb.libsonnet').get()
    else if 'decision_tree' == _Utils.network then
        (import 'types/decision_tree.libsonnet').get()
    else if 'random_forest' == _Utils.network then
        (import 'types/random_forest.libsonnet').get()
    else if 'logistic' == _Utils.network then
        (import 'types/logistic.libsonnet').get()
    else if 'gradient_boosting' == _Utils.network then
        (import 'types/gradient_boosting.libsonnet').get()
    else []
)
