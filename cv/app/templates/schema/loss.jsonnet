// @file loss.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-10 18:22

local _reduction_ = [
    {
        name: 'mean',
        value: 'mean',
    },
    {
        name: 'sum',
        value: 'sum',
    },
    {
        name: 'none',
        value: 'none',
    },
];

{
    loss: {
        type: 'object',
        name: 'Loss',
        description: |||
            todo
        |||,

        loss_type: {
            type: 'enum',
            name: 'Loss Type',
            description: |||
                todo
            |||,

            items: {
                type: 'string',
                values: [
                    {
                        name: 'CE Loss',
                        value: 'ce_loss',
                        ref: 'loss.loss_weights.ce_loss',
                    },
                    {
                        name: 'Soft CE Loss',
                        value: 'soft_ce_loss',
                        ref: 'loss.loss_weights.soft_ce_loss',
                    },
                ],
            },
            default: 0,
        },

        loss_weights: {
            type: 'object',
            description: @'Loss Weight',

            ce_loss: {
                type: 'object',
                name: 'CE Loss',

                ce_loss: {
                    type: 'float',
                    name: 'CE Loss Weight',
                    default: 1,
                    ref: 'loss.params.ce_loss',
                },
            },

            soft_ce_loss: {
                type: 'object',
                name: 'Soft CE Loss',

                sof_ce_loss: {
                    type: 'float',
                    name: 'Soft CE Loss Weight',
                    default: 1,
                    ref: 'loss.params.soft_ce_loss',
                },
            },
        },

        params: {
            type: 'object',
            description: @'Loss Function Parameters',

            ce_loss: {
                type: 'object',
                name: 'CE Loss Parameters',

                reduction: {
                    type: 'enum',
                    name: 'Reduction',

                    items: {
                        type: 'string',
                        values: _reduction_,
                    },
                    default: 0,
                },

                ignore_index: {
                    type: 'float',
                    name: 'Value Ignore',

                    default: -1,
                },
            },

            soft_ce_loss: {
                type: 'object',
                name: 'Soft CE Loss Parameters',

                reduction: {
                    type: 'enum',
                    name: 'Reduction',

                    items: {
                        type: 'string',
                        values: [
                            {
                                name: 'batchmean',
                                value: 'batchmean',
                            },
                        ] + _reduction_,
                    },
                    default: 0,
                },

            },
        },
    },
}
