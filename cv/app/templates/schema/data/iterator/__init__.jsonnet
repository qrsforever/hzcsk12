// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 14:46

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: [
        {
            type: 'H',
            objs: [
                _Utils.int('solver.display_iter', 'Display Iters', def=20, ddd=true),
                _Utils.int('solver.save_iters', 'Save Iters', def=200, ddd=true),
                _Utils.int('solver.test_interval', 'Test Iters', def=2000, ddd=true),
            ],
        },
        {
            type: 'H',
            objs: [
                _Utils.int('train.batch_size', 'Train Batch Size', def=32, ddd=true),
                _Utils.int('val.batch_size', 'Val Batch Size', def=32, ddd=true),
                _Utils.int('test.batch_size', 'Test Batch Size', def=32, ddd=true),
            ],
        },
    ],
}
