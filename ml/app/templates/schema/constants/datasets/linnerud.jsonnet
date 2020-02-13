{
    task: 'regressor',
    method: 'sklearn_wrapper',
    data: {
        num_samples: 20,
        num_features: 3,
        data_path: 'load_linnerud',
        sampling: {
            test_size: 0.25,
            random_state: 1,
            shuffle: true,
        },
    },
    model: {
        type: 'random_forest',
        random_forest: {
            n_estimators: 10,
            criterion: 'mse',
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_weight_fraction_leaf: 0.0,
            max_features: 0.3,
            max_leaf_nodes: 10,
            min_impurity_decrease: 0.0,
            n_jobs: 1,
            warm_start: false,
            verbose: 0,
            oob_score: false,
            bootstrap: false,
        },
    },
}
