// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 23:05

local _Utils = import '../utils/helper.libsonnet';

(import 'networks/__init__.jsonnet').get()
+
[
    {
        type: 'accordion',
        objs: (
                  if _Utils.method == 'single_shot_detector' then [
                      {
                          name: { en: 'SSD', cn: self.en },
                          type: '_ignore_',
                          objs: [
                              {
                                  type: 'H',
                                  objs: [
                                      {
                                          _id_: 'anchor.anchor_method',
                                          name: { en: 'Anchor Method', cn: self.en },
                                          type: 'string-enum',
                                          objs: [
                                              {
                                                  name: { en: 'ssd', cn: self.en },
                                                  value: 'ssd',
                                              },
                                              {
                                                  name: { en: 'retina', cn: self.en },
                                                  value: 'retina',
                                              },
                                              {
                                                  name: { en: 'naive', cn: self.en },
                                                  value: 'naive',
                                              },
                                          ],
                                          default: self.objs[0].value,
                                          readonly: true,
                                      },
                                      _Utils.float('anchor.iou_threshold', 'IOU Threshold', def=0.5, ddd=true),
                                  ],
                              },
                              {
                                  type: 'H',
                                  objs: [
                                      _Utils.float('res.nms.max_threshold', 'NMS Threshold', def=0.5, ddd=true),
                                      _Utils.float('res.val_conf_thre', 'Score Threshold', def=0.05, ddd=true),
                                      _Utils.float('res.vis_conf_thre', 'Visual Threshold', def=0.5, ddd=true),
                                  ],
                              },
                              {
                                  type: 'H',
                                  objs: [
                                      _Utils.int('res.nms.pre_nms', 'NMS Pre', def=1000, ddd=true),
                                      _Utils.int('res.max_per_image', 'Max Per Image', def=200, ddd=true),
                                      _Utils.int('res.cls_keep_num', 'Max Per Class', def=20, ddd=true),
                                  ],
                              },
                          ],
                      },
                  ] else []
              )
              + [
                  {
                      name: { en: 'Metrics', cn: self.en },
                      type: '_ignore_',
                      objs: [
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
                                                  _Utils.bool('metrics.train_speed', 'Speed', def=false),
                                                  _Utils.bool('metrics.train_lr', 'Learn Rate', def=false),
                                              ],
                                          },
                                      ],
                                  },
                                  {
                                      name: { en: 'Val', cn: self.en },
                                      type: '_ignore_',
                                      objs: [
                                          {
                                              type: 'H',
                                              objs: [
                                                  _Utils.bool('metrics.val_speed', 'Speed', def=false),
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
                                                  _Utils.bool('metrics.true_pred_images', 'Top 10 Images', def=true),
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
                                                  _Utils.bool('metrics.fm', 'Feature Maps', def=false),
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
                              ],
                          },
                      ],
                  },
              ]
              // + (
              //     if std.startsWith(_Utils.network, 'custom_') then [
              //         {
              //             name: { en: 'Custom', cn: self.en },
              //             type: '_ignore_',
              //             objs: [
              //                 {
              //                     _id_: 'network.net_def',
              //                     type: 'iframe',
              //                     html: '',
              //                     width: 800,
              //                     height: 400,
              //                 },
              //             ],
              //         },
              //     ] else []
              // )
              + [
                  {
                      name: { en: 'Details', cn: self.en },
                      type: '_ignore_',
                      objs: (import 'details/__init__.jsonnet').get(),
                  },
              ],
    },
]
