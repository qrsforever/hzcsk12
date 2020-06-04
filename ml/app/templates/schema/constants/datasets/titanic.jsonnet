{
    task: 'classifier',
    method: 'sklearn_wrapper',
    data: {
        num_samples: 891,
        num_features: 11,
        num_classes: 2,
        data_path: '/datasets/titanic',
        sampling: {
            test_size: 0.25,
            random_state: 1,
            shuffle: true,
        },
    },
    model: {
        type: 'decision_tree',
    },
    _k12: {
        detail: {
            description: |||
                https://www.kaggle.com/c/titanic/
            |||,
        },
    },
    metrics: {
        dtreeviz: {
            target_name: 'titanic',
        },
    },
}
