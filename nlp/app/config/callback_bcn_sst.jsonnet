{
    "dataset_reader":{
        "type": "sst_tokens",
        "use_subtrees": true,
        "granularity": "5-class"
    },
    "validation_dataset_reader":{
        "type": "sst_tokens",
        "use_subtrees": false,
        "granularity": "5-class"
    },
    "train_data_path": "/data/datasets/nlp/sst/train.txt",
    "validation_data_path": "/data/datasets/nlp/sst/dev.txt",
    "test_data_path": "/data/datasets/nlp/sst/test.txt",
    "model": {
        "type": "bcn",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 200,
                    "pretrained_file": "/data/datasets/nlp/glove/glove.6B.200d.txt.gz",
                    "trainable": false
                }
            }
        },
        "embedding_dropout": 0.25,
        "pre_encode_feedforward": {
            "input_dim": 200,
            "num_layers": 1,
            "hidden_dims": [200],
            "activations": ["relu"],
            "dropout": [0.25]
        },
        "encoder": {
            "type": "lstm",
            "input_size": 200,
            "hidden_size": 200,
            "num_layers": 1,
            "bidirectional": true
        },
        "integrator": {
            "type": "lstm",
            "input_size": 1200,
            "hidden_size": 200,
            "num_layers": 1,
            "bidirectional": true
        },
        "integrator_dropout": 0.1,
        "output_layer": {
            "input_dim": 1600,
            "num_layers": 3,
            "output_dims": [800, 400, 5],
            "pool_sizes": 4,
            "dropout": [0.2, 0.3, 0.0]
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size" : 2
    },
    "trainer": {
        "type": "callback",
        "num_epochs": 2,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "callbacks": [
            {"type": "track_metrics", "patience": 5},
            "validate",
            {"type": "log_to_visdom", "server_port": 8150}
        ],
        "cuda_device": 0
    }
}
