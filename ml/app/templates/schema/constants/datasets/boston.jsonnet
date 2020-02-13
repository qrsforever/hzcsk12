{
    task: 'regressor',
    method: 'sklearn_wrapper',
    data: {
        num_samples: 506,
        num_features: 13,
        data_path: 'load_boston',
        sampling: {
            test_size: 0.25,
            random_state: 1,
            shuffle: true,
        },
    },
    model: {
        type: 'knn',
        knn: {
            n_neighbors: 5,
            weights: 'uniform',
            algorithm: 'auto',
            leaf_size: 30,
            p: 2,
            n_jobs: 1,
        },
    },
}
