// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 14:46

local _Utils = import '../../utils/helper.libsonnet';

{
    get():: {
        type: '_ignore_',
        objs: [
            {
                type: 'H',
                objs: [
                    _Utils.int('solver.display_iter', 'Display Iters', def=200, ddd=true),
                    _Utils.int('solver.save_iters', 'Save Iters', def=2000, ddd=true),
                    _Utils.int('solver.test_interval', 'Test Iters', def=2000, ddd=true),
                ],
            },
            {
                name: { en: 'Phase', cn: self.en },
                type: 'navigation',
                objs: [
                    {
                        name: { en: 'Train', cn: self.en },
                        type: '_ignore_',
                        objs: (import 'common.libsonnet').get('train'),
                    },
                    {
                        name: { en: 'Validation', cn: self.en },
                        type: '_ignore_',
                        objs: (import 'common.libsonnet').get('val'),
                    },
                    {
                        name: { en: 'Test', cn: self.en },
                        type: '_ignore_',
                        objs: (import 'common.libsonnet').get('test'),
                    },
                ],
            },
        ],
    },
}
