// @file evaluate.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-22 18:01

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.bool('metrics.evaluate.absrel', 'absrel'),
            _Utils.bool('metrics.evaluate.mse', 'MSE'),
            _Utils.bool('metrics.evaluate.rmse', 'RMSE'),
            _Utils.bool('metrics.evaluate.mae', 'MAE'),
        ],
    },
]
