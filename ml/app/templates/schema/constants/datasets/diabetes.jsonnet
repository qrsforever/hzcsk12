{
    task: 'regressor',
    method: 'sklearn_wrapper',
    data: {
        num_samples: 442,
        num_features: 10,
        data_path: 'load_diabetes',
        sampling: {
            test_size: 0.25,
            random_state: 1,
            shuffle: true,
        },
    },
    model: {
        type: 'svr',
        svr: {
            C: 1.0,
            kernel: 'linear',
            degree: 3,
            gamma: '0.2',
            coef0: 0.0,
            shrinking: true,
            tol: 1e-3,
            cache_size: 200.0,
            max_iter: 1000,
        },
    },
}
