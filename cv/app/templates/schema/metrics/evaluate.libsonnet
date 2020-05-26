// @file evaluate.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-26 20:18

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.bool('metrics.confusion_matrix', 'Confusion Matrix', def=true),
            _Utils.bool('metrics.top10_images', 'Top 10 Images', def=false),
            _Utils.bool('metrics.top10_errors', 'Top 10 Errors', def=false),
            _Utils.bool('metrics.precision',
                        'Precision',
                        def=true,
                        tips='tp / (tp + fp)'),
            _Utils.bool('metrics.recall',
                        'Recall',
                        def=true,
                        tips='tp / (tp + fn)'),
            _Utils.bool('metrics.fscore',
                        'Fscore',
                        def=false,
                        tips='harmonic mean of precision and recall'),
            _Utils.bool('metrics.model_autograd', 'Model AutoGrad', def=false),
            _Utils.bool('metrics.model_graph', 'Model Graph', def=false),
            _Utils.bool('metrics.feature_maps', 'Feature Maps', def=false),
            _Utils.bool('metrics.filters_maps', 'Filters Maps', def=false),
            _Utils.bool('metrics.vbp', 'Saliency Maps', def=false),
            _Utils.bool('metrics.gbp', 'Guided BP', def=false),
            _Utils.bool('metrics.deconv', 'Deconvnet', def=false),
            _Utils.bool('metrics.gcam',
                        'G-CAM',
                        def=false,
                        tips='only for resnet and vgg now'),
            _Utils.bool('metrics.ggcam',
                        'Guided G-CAM',
                        def=false,
                        tips='only for resnet and vgg now'),
        ],
    },
]
