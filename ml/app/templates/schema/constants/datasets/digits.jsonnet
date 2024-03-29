{
    task: 'classifier',
    method: 'sklearn_wrapper',
    data: {
        num_samples: 1797,
        num_features: 64,
        num_classes: 10,
        data_path: 'load_digits',
        sampling: {
            test_size: 0.25,
            random_state: 1,
            shuffle: true,
        },
    },
    model: {
        type: 'knn',
        knn: {

        },
        svc: {
            C: 1.0,
            kernel: 'rbf',
            degree: 3,
            gamma: '0.2',
            coef0: 0.0,
            shrinking: true,
            probability: false,
            tol: 1e-3,
            cache_size: 200.0,
            max_iter: 1000,
            decision_function_shape: 'ovr',
        },
        random_forest: {
            n_estimators: 10,
            criterion: 'gini',
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_weight_fraction_leaf: 0.0,
            max_features: 0.2,
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
