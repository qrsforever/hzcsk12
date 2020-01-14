// @file helper.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 13:31

local _network_maps = {
    base_model: {
        method: 'image_classifier',
        name: { en: 'base', cn: self.en },
    },
    cls_model: {
        method: 'image_classifier',
        name: { en: 'cls', cn: self.en },
    },
    distill: {
        method: 'image_classifier',
        name: { en: 'distill', cn: self.en },
    },
    vgg16_ssd300: {
        method: 'single_shot_detector',
        name: { en: 'ssd300', cn: self.en },
    },
    vgg16_ssd512: {
        method: 'single_shot_detector',
        name: { en: 'ssd512', cn: self.en },
    },
    lffdv2: {
        method: 'single_shot_detector',
        name: { en: 'lffdv2', cn: self.en },
    },
    faster_rcnn: {
        method: 'faster_rcnn',
        name: { en: 'faster rcnn', cn: self.en },
    },
    darknet_yolov3: {
        method: 'yolov3',
        name: { en: 'yolov3', cn: self.en },
    },
};

{
    version:: '0.0.1b',
    task:: std.extVar('task'),
    network:: std.extVar('network'),
    method:: _network_maps[$.network].method,
    network_name:: _network_maps[$.network].name,
    dataset_name:: std.extVar('dataset_name'),

    get_value(obj, keystr, def)::
        if std.type(obj) == 'object' && std.length(keystr) > 1
        then
            local keys = std.splitLimit(keystr, '.', 1);
            if std.objectHas(obj, keys[0])
            then
                if std.length(keys) > 1
                then
                    $.get_value(obj[keys[0]], keys[1], def)
                else
                    obj[keys[0]]
            else
                def
        else
            def,

    // get default from constants dataset configs
    get_default_value(keystr, def)::
        if std.objectHas($.datasets, $.dataset_name)
        then
            $.get_value($.datasets[$.dataset_name], keystr, def)
        else
            def,

    // default dataset, can set default value
    datasets:: {
        [if $.dataset_name == 'mnist' then 'mnist']: import '../constants/datasets/mnist.jsonnet',
        [if $.dataset_name == 'cifar10' then 'cifar10']: import '../constants/datasets/cifar10.jsonnet',
        [if $.dataset_name == 'VOC07+12_DET' then 'VOC07+12_DET']: import '../constants/datasets/VOC07+12_DET.jsonnet',
    },

    // basic type node generator function
    bool(id, en, cn='', def=false, ddd=false, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'bool',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    int(id, en, cn='', def=0, ddd=false, tips='', min=-1, max=-1, width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'int',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if min > 0 then 'min']: min,
        [if max > 0 then 'max']: max,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    float(id, en, cn='', def=0, ddd=false, tips='', min=-1, max=-1, width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'float',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if min > 0 then 'min']: min,
        [if max > 0 then 'max']: max,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    string(id, en, cn='', def='', ddd=false, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'string',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    text(id, en, cn='', def='', ddd=false, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'text',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    intarray(id, en, cn='', def=[], ddd=false, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'int-array',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    floatarray(id, en, cn='', def=[], ddd=false, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'float-array',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    stringarray(id, en, cn='', def='', ddd=false, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'string-array',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },
}
