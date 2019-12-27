// @file constants.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-25 23:33

// jsonnet --tla-str dataset_path="/data/" --tla-str dataset_name="xxx"
// jsonnet --ext-str dataset_path="/data/" --ext-str dataset_name="xxx"
// function(dataset_path='', dataset_name='') {
//     dataset_path:: if std.length(dataset_path) == 0 then '/data/datasets/nlp' else dataset_path,
//     dataset_name:: if std.length(dataset_name) == 0 then 'default' else dataset_name,
// }

{
    dataset_path:: std.extVar('dataset_path'),
    dataset_name:: std.extVar('dataset_name'),
}
