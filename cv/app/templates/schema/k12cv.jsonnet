// @file k12cv.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 17:59

{
    description: |||
        k12cv configure test
    |||,

    type: 'page',
    objs: [
        {
            name: { en: 'data', cn: self.en },
            type: 'tab',
            objs: [
                {
                    name: { en: 'Dataset', cn: self.en },
                    type: 'accordion',
                    objs: [
                    ],
                },
                {
                    name: { en: 'Transform', cn: self.en },
                    type: 'accordion',
                    objs: [
                        {
                            name: { en: 'Phase', cn: self.en },
                            type: 'object-trigger',
                            objs: [
                                (import 'trans/trans.libsonnet').get('train'),
                                (import 'trans/trans.libsonnet').get('validate'),
                                (import 'trans/trans.libsonnet').get('test'),
                            ],
                        },
                    ],  // objs
                },
            ],  // objs
        },
        {
            name: { en: 'model', cn: self.en },
            type: 'tab',
            objs: [],
        },
        {
            name: { en: 'hypes', cn: self.en },
            type: 'tab',
            objs: [
                import 'lr/lr.jsonnet',
                import 'optimizer/optimizer.jsonnet',
                import 'loss/loss.jsonnet',
                {
                    name: { en: 'Iterator', cn: self.en },
                    type: 'accordion',
                    objs: [
                        {
                            _id_: 'solver.max_iters',
                            name: { en: 'Max Iterator', cn: self.en },
                            type: 'int',
                            default: 20000,
                        },
                        {
                            _id_: 'solver.display_iter',
                            name: { en: 'Display Iter', cn: self.en },
                            type: 'int',
                            default: 200,
                        },
                        {
                            _id_: 'solver.save_iters',
                            name: { en: 'Save Iter', cn: self.en },
                            type: 'int',
                            default: 2000,
                        },
                        {
                            _id_: 'solver.test_interval',
                            name: { en: 'test interval', cn: self.en },
                            type: 'int',
                            default: 2000,
                        },
                    ],  // objs
                },
            ],  // objs
        },
    ],  // objs
}
