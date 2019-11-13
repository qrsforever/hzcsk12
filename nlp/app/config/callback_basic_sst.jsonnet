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
        "type": "lstm_classifier",
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
        "seq2vec_encoder": {
            "type": "lstm",
            "input_size": 200,
            "hidden_size": 512,
            "num_layers": 2,
            "batch_first": true
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size" : 32
    },
    "trainer": {
        "type": "callback",
        "num_epochs": 2,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "callbacks": [
            {"type": "track_metrics", "patience": 1},
            "validate",
            {"type": "log_to_visdom", "server_port": 8140}
        ],
        "cuda_device": 0
    }
}
