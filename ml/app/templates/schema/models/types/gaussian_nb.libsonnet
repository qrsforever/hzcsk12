// @file gaussian_nb.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-13 22:32


local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.float('model.gaussian_nb.var_smoothing', 'Smoothing', def=1e-9, ddd=true),
            ],
        },
    ],
}
