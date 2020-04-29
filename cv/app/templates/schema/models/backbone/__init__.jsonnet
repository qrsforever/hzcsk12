// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 18:13

local _Utils = import '../../utils/helper.libsonnet';
local _IsCustom = std.startsWith(_Utils.network, 'custom_');

{
    get()::
        if _Utils.method == 'image_classifier' then {
            _id_: 'network.backbone',
            name: { en: 'Backbone', cn: self.en },
            type: 'string-enum',
            objs: (import 'vgg.libsonnet').get() +
                  (import 'resnet.libsonnet').get() +
                  (import 'alexnet.libsonnet').get() +
                  (if _IsCustom then [
                       { name: { en: 'custom', cn: self.en }, value: 'custom' },
                   ] else []),
            default: if _IsCustom then 'custom' else _Utils.get_default_value(self._id_, 'resnet50'),
            readonly: _IsCustom,
        } else if _Utils.method == 'single_shot_detector' then {
            _id_: 'network.backbone',
            name: { en: 'Backbone', cn: self.en },
            type: 'string-enum',
            objs: [
                { name: { en: 'vgg16', cn: self.en }, value: 'vgg16' },
                { name: { en: 'custom', cn: self.en }, value: 'custom' },
            ],
            default: if _IsCustom then 'custom' else 'vgg16',
            readonly: true,
        } else {
        },
}
