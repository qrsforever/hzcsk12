// @file sst.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 20:33

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
    train_data_path: '/data/datasets/nlp/sst/train.txt',
    validation_data_path: '/data/datasets/nlp/sst/dev.txt',
    test_data_path: '/data/datasets/nlp/sst/test.txt',
}
