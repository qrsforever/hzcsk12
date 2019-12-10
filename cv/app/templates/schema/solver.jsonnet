// @file solver.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-10 21:20

local _gamma_ = {
    type: 'float',
    name: 'Gamma',
    default: 0.10,
};

{
    solver: {
        type: 'object',
        name: 'Solver',
        description: |||
            todo finish
        |||,

        display_iter: {
            type: 'int',
            name: 'Display Iterator',
            default: 100,
        },

        save_iters: {
            type: 'int',
            name: 'Save Iterator',
            default: 10000,
        },

        test_interval: {
            type: 'int',
            name: 'Test Interval',
            default: 4000,
        },

        max_iters: {
            type: 'int',
            name: 'Max Iterators',
            default: 28000,
        },

        lr: {
            type: 'object',
            name: 'Learn Rate',

            base_lr: {
                type: 'float',
                name: 'Base Learn Rate',
                default: 0.01,
            },

            metric: {
                type: 'enum',
                name: 'Metric',
                items: {
                    type: 'string',
                    multiple: false,
                    values: [
                        {
                            name: 'iters',
                            value: 'epoch',
                        },
                        {
                            name: 'iters',
                            value: 'iters',
                        },
                    ],
                },
                default: 0,
            },

            lr_policy: {
                type: 'enum',
                name: 'Learn Rate Policy',
                items: {
                    type: 'string',
                    multiple: false,
                    values: [
                        {
                            name: 'step',
                            value: 'step',
                            ref: 'solver.step',
                        },
                        {
                            name: 'multistep',
                            value: 'multistep',
                            ref: 'solver.lr.multistep',

                        },
                        {
                            name: 'lambda_poly',
                            value: 'lambda_poly',
                            ref: 'solver.lr.lambda_poly',
                        },
                    ],
                },
                default: 0,
            },

            step: {
                type: 'object',
                name: 'Step Parameters',
                gamma: _gamma_,
                step_size: {
                    type: 'int',
                    name: 'Step Size',
                    default: 50,
                },
            },

            multistep: {
                type: 'object',
                name: 'Multistep Parameters',
                gamma: _gamma_,
                stepvalue: {
                    type: 'int-array',
                    name: 'Multistep Size',
                    minitem: 2,
                    default: [90, 120],
                },
            },

            lambda_poly: {
                type: 'object',
                name: 'Lambda Policy Parameters',
                power: {
                    type: 'float',
                    name: 'power',
                    default: 0.90,
                },
            },
        },  // lr

        optim: {
            type: 'object',
            name: 'Optimize',
            optim_method: {
                type: 'enum',
                name: 'Optim Method',
                items: {
                    type: 'string',
                    multiple: false,
                    values: [
                        {
                            name: 'sgd',
                            value: 'SGD',
                            ref: 'solver.optim.sgd',
                        },
                        {
                            name: 'adam',
                            value: 'Adam',
                            ref: 'solver.optim.adam',
                        },
                    ],
                },
                default: 0,
            },
            sgd: {
                type: 'object',
                name: 'SGD Parameters',
                weight_decay: {
                    type: 'float',
                    default: 0.0001,
                    range: '(0, 1]',
                },
                momentum: {
                    type: 'float',
                    default: 0.9,
                    range: '(0, 1]',
                },
                nesterov: {
                    type: 'bool',
                    default: false,
                },
            },
            adam: {
                type: 'object',
                name: 'Adam Parameters',
                weight_decay: {
                    type: 'float',
                    default: 0.0001,
                    range: '(0, 1]',
                },
                betas: {
                    type: 'float-array',
                    name: 'Betas',
                    range: '(0, 1]',
                    minitem: 2,
                    maxitem: 2,
                    default: [0.5, 0.999],
                },
                eps: {
                    type: 'float',
                    name: 'EPS',
                    default: 1e-08,
                    range: '(0, 1]',
                },
            },
        },  // optim
    },
}
