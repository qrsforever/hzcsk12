{
    task: 'regressor',
    method: 'sklearn_wrapper',
    data: {
        num_samples: 878049,
        num_features: 9,
        num_classes: 39,
        data_path: '/datasets/sf-crime',
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
        knn: {
            n_neighbors: 39,
            weights: 'uniform',
            algorithm: 'auto',
            leaf_size: 30,
            p: 2,
            n_jobs: 1,
        },
    },
    _k12: {
        detail: {
            description: |||
                1. Dates - timestamp of the crime incident
                2. Category - category of the crime incident
                3. Descript - detailed description of the crime incident
                4. DayOfWeek - the day of the week
                5. PdDistrict - name of the Police Department District
                6. Resolution - how the crime incident was resolved
                7. Address - the approximate street address of the crime incident
                8. X - Longitude
                9. Y - Latitude
            |||,
        },
    },
}
