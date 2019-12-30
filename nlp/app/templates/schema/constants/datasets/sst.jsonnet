// @file sst.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 23:33

{
    get(dataset_path):: {
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
        train_data_path: dataset_path + '/sst/train.txt',
        validation_data_path: dataset_path + '/sst/dev.txt',
        test_data_path: dataset_path + '/sst/test.txt',
        model: {
            type: 'basic_classifier',
            text_field_embedder: {
                token_embedders: {
                    tokens: {
                        type: 'embedding',
                        embedding_dim: 200,
                        pretrained_file: dataset_path + '/glove/glove.6B.200d.txt.gz',
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
        iterator: {
            type: 'bucket',
            sorting_keys: [['tokens', 'num_tokens']],
            batch_size: 32,
        },
        trainer: {
            num_epochs: 2,
            patience: 1,
            grad_norm: 5.0,
            validation_metric: '+accuracy',
            cuda_device: 0,
            optimizer: {
                type: 'adam',
                lr: 0.001,
            },
        },
    },
}
