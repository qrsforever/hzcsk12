{
    task: 'regressor',
    method: 'sklearn_wrapper',
    data: {
        num_samples: 1459,
        num_features: 80,
        num_classes: 1,
        data_path: '/datasets/house-prices',
        sampling: {
            test_size: 0.25,
            random_state: 1,
            shuffle: true,
        },
    },
    model: {
        type: 'gradient_boosting',
        gradient_boosting: {
            n_estimators: 200,
        },
    },
    _k12: {
        detail: {
            description: |||
                https://www.kaggle.com/c/house-prices-advanced-regression-techniques
            |||,
        },
    },
}
