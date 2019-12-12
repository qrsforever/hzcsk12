// @file k12cv.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 17:59

{
    description: |||
        k12cv configure
    |||,

    data: {

        dataset: {

        },

        loader: {
            train: {
                aug_trans: {
                    shuffle_trans_seq: {
                    },
                    trans_seq: {
                    },
                },
                data_trans: {
                },
            },
            val: {
            },
            test: {
            },
        },
    },

    model: {

    },

    hypes: {
        lr: import 'lr/lr.jsonnet',
        optimizer: import 'optimizer/optimizer.jsonnet',
        loss: import 'loss/loss.jsonnet',
        iterator: {
            name: { en: 'Iterator', cn: self.en },
            type: 'object',
            objs: ['max_iters', 'display_iter', 'save_iters', 'test_interval'],
            max_iters: {
                _id_: 'solver.max_iters',
                name: { en: 'Max Iterator', cn: self.en },
                type: 'int',
                default: 20000,
            },
            display_iter: {
                _id_: 'solver.display_iter',
                name: { en: 'Display Iter', cn: self.en },
                type: 'int',
                default: 200,
            },
            save_iters: {
                _id_: 'solver.save_iters',
                name: { en: 'Save Iter', cn: self.en },
                type: 'int',
                default: 2000,
            },
            test_interval: {
                _id_: 'solver.test_interval',
                name: { en: 'test interval', cn: self.en },
                type: 'int',
                default: 2000,
            },
        },
    },
}
