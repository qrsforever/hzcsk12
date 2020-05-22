// @file conll2003.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-22 16:16

{
    dataset_reader: {
        type: 'conll2003',
        tag_label: 'ner',
        coding_scheme: 'BIOUL',
        token_indexers: {
            tokens: {
                type: 'single_id',
                lowercase_tokens: true,
            },
            token_characters: {
                type: 'characters',
                min_padding_length: 3,
            },
        },
    },
    train_data_path: '/datasets/conll2003/train.txt',
    validation_data_path: '/datasets/conll2003/sst/dev.txt',
    test_data_path: '/datasets/conll2003/test.txt',
    model: {
        type: 'crf_tagger',
        label_encoding: 'BIOUL',
        constrain_crf_decoding: true,
        calculate_span_f1: true,
        dropout: 0.5,
        include_start_end_transitions: false,
        text_field_embedder: {
            token_embedders: {
                tokens: {
                    type: 'embedding',
                    embedding_dim: 50,
                    pretrained_file: '/data/pretrained/nlp/glove/glove.6B.50d.txt.gz',
                    trainable: true,
                },
                token_characters: {
                    type: 'character_encoding',
                    embedding: {
                        embedding_dim: 16,
                    },
                    encoder: {
                        type: 'cnn',
                        embedding_dim: 16,
                        num_filters: 128,
                        ngram_filter_sizes: [3],
                        conv_layer_activation: 'relu',
                    },
                },
            },
        },
        encoder: {
            type: 'lstm',
            input_size: 50 + 128,
            hidden_size: 200,
            num_layers: 2,
            dropout: 0.5,
            bidirectional: true,
        },
    },
    data_loader: {
        batch_size: 64,
    },
    trainer: {
        optimizer: {
            type: 'adam',
            lr: 0.001,
        },
        checkpointer: {
            num_serialized_models_to_keep: 3,
        },
        validation_metric: '+f1-measure-overall',
        num_epochs: 75,
        grad_norm: 5.0,
        patience: 25,
        cuda_device: 0,
    },
}
