// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 18:13

local _Utils = import '../../utils/helper.libsonnet';

{
    get()::
        if _Utils.method == 'image_classifier' then {
            _id_: 'network.backbone',
            name: { en: 'Backbone', cn: self.en },
            type: 'string-enum',
            objs: (import 'vgg.libsonnet').get() +
                  (import 'resnet.libsonnet').get(),
            default: _Utils.get_default_value(self._id_, 'vgg16'),
        } else if _Utils.method == 'single_shot_detector' then {
            _id_: 'network.backbone',
            name: { en: 'Backbone', cn: self.en },
            type: 'string-enum',
            objs: [
                { name: { en: 'vgg16', cn: self.en }, value: 'vgg16' },
                { name: { en: 'custom', cn: self.en }, value: 'custom' },
            ],
            default: if std.startsWith(_Utils.network, 'custom_') then 'custom' else 'vgg16',
            readonly: true,
        } else {
        },
}
