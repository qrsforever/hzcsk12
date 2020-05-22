// @file sst.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 23:33

{
    dataset_reader: {
        type: 'sst_tokens',
        use_subtrees: true,
        granularity: '5-class',
    },
    validation_dataset_reader: {
        type: 'sst_tokens',
        use_subtrees: false,
        granularity: '5-class',
    },
    train_data_path: '/datasets/sst/train.txt',
    validation_data_path: '/datasets/sst/dev.txt',
    test_data_path: '/datasets/sst/test.txt',
    model: {
        type: 'basic_classifier',
        text_field_embedder: {
            token_embedders: {
                tokens: {
                    type: 'embedding',
                    embedding_dim: 200,
                    pretrained_file: '/pretrained/glove/glove.6B.200d.txt.gz',
                    trainable: false,
                },
            },
        },
        seq2vec_encoder: {
            type: 'lstm',
            input_size: 200,
            hidden_size: 512,
            num_layers: 2,
            batch_first: true,
        },
    },
    data_loader: {
        batch_sampler: {
            type: 'bucket',
            batch_size: 32,
        },

    },
    trainer: {
        num_epochs: 1000,
        patience: 10,
        grad_norm: 5.0,
        summary_interval: 100,
        validation_metric: '+accuracy',
        distributed: false,
        cuda_device: 0,
        optimizer: {
            type: 'adam',
            lr: 0.001,
        },
    },
    _k12: {
        validation_dataset_reader: true,
        validation_iterator: true,
    },
}
