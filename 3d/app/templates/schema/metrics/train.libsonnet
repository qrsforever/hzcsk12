// @file train.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-22 18:01

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.bool('metrics.train.loss', 'loss', def=true, readonly=true),
            _Utils.bool('metrics.train.mse', 'MSE'),
            _Utils.bool('metrics.train.rmse', 'RMSE'),
            _Utils.bool('metrics.train.mae', 'MAE'),
            _Utils.bool('metrics.train.irmse', 'iRMSE'),
            _Utils.bool('metrics.train.imae', 'iMAE'),
            _Utils.bool('metrics.train.absrel', 'absrel'),
            _Utils.bool('metrics.train.delta1', 'delta1'),
            _Utils.bool('metrics.train.delta2', 'delta2'),
            _Utils.bool('metrics.train.delta3', 'delta3'),
            _Utils.bool('metrics.train.speed', 'speed'),
        ],
    },
]
