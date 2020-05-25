// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-05-25 15:35

local _Utils = import '../utils/helper.libsonnet';

[
    {
        name: { en: 'Phase', cn: self.en },
        type: 'navigation',
        objs: [
            {
                name: { en: 'Train', cn: self.en },
                type: '_ignore_',
                objs: [
                    {
                        type: 'H',
                        objs: [
                            _Utils.bool('metrics.raw_vs_aug', 'Raw & Aug', def=false),
                            _Utils.bool('metrics.train_speed', 'Train Speed', def=false),
                            _Utils.bool('metrics.train_lr', 'Learn Rate', def=false),
                            _Utils.bool('metrics.val_speed', 'Valid Speed', def=false),
                        ],
                    },
                ],
            },
            {
                name: { en: 'Evaluate', cn: self.en },
                type: '_ignore_',
                objs: [
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
                ],
            },
            {
                name: { en: 'Pridict', cn: self.en },
                type: '_ignore_',
                objs: [
                    {
                        type: 'H',
                        objs: [
                        ],
                    },
                ],
            },
        ],
    },
]

// + [
// ]



