// @file helper.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 13:31

local _network_maps = {
    custom_base: {
        method: 'image_classifier',
        backbone: 'custom',
    },
    vgg16_ssd300: {
        method: 'single_shot_detector',
        backbone: 'vgg16',
    },
    vgg16_ssd512: {
        method: 'single_shot_detector',
        backbone: 'vgg16',
    },
    custom_ssd300: {
        method: 'single_shot_detector',
        backbone: 'custom',
    },
    custom_ssd512: {
        method: 'single_shot_detector',
        backbone: 'custom',
    },
    lffdv2: {
        method: 'single_shot_detector',
        backbone: 'lffdv2',
    },
    faster_rcnn: {
        method: 'faster_rcnn',
        backbone: 'faster rcnn',
    },
    darknet_yolov3: {
        method: 'yolov3',
        backbone: 'yolov3',
    },
    pix2pix: {
        method: 'image_translator',
        backbone: 'none',
    },
};

{
    levelid:: std.extVar('levelid'),
    debug:: std.extVar('debug'),
    net_ip:: std.extVar('net_ip'),
    task:: std.extVar('task'),
    num_cpu:: std.extVar('num_cpu'),
    num_gpu:: std.extVar('num_gpu'),
    network_:: std.extVar('network'),
    network:: if std.objectHas(_network_maps, $.network_) then $.network_ else 'base_model',
    method:: if std.objectHas(_network_maps, $.network_) then _network_maps[$.network_].method else 'image_classifier',
    backbone:: if std.objectHas(_network_maps, $.network_) then _network_maps[$.network_].backbone else $.network_,
    dataset_name:: std.extVar('dataset_name'),
    dataset_info:: std.extVar('dataset_info'),
    dataset_root:: '/datasets/cv/' + $.dataset_name + '/',

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
        if std.length($.dataset_constants) > 0
        then
            $.get_value($.dataset_constants, keystr, def)
        else
            def,

    // default dataset, can set default value
    dataset_constants:: if std.length($.dataset_info) > 3 then std.parseJson($.dataset_info)
    else if $.dataset_name == 'rclothing4' then import '../constants/datasets/rclothing4.jsonnet'
    else if $.dataset_name == 'rclothing16' then import '../constants/datasets/rclothing16.jsonnet'
    else if $.dataset_name == 'rcnfood12' then import '../constants/datasets/rcnfood12.jsonnet'
    else if $.dataset_name == 'rcnfood35' then import '../constants/datasets/rcnfood35.jsonnet'
    else if $.dataset_name == 'ranimals6' then import '../constants/datasets/ranimals6.jsonnet'
    else if $.dataset_name == 'ranimals25' then import '../constants/datasets/ranimals25.jsonnet'
    else if $.dataset_name == 'roffice6' then import '../constants/datasets/roffice6.jsonnet'
    else if $.dataset_name == 'rflowers' then import '../constants/datasets/flowers.jsonnet'
    else if $.dataset_name == 'rflowers5' then import '../constants/datasets/flowers5.jsonnet'
    else if $.dataset_name == 'rfruits' then import '../constants/datasets/rfruits.jsonnet'
    else if $.dataset_name == 'rmnist' then import '../constants/datasets/mnist.jsonnet'
    else if $.dataset_name == 'cleaner_robot' then import '../constants/datasets/cleaner_robot.jsonnet'
    else if $.dataset_name == 'rcifar10' then import '../constants/datasets/cifar10.jsonnet'
    else if $.dataset_name == 'rDogsVsCats' then import '../constants/datasets/dogsVsCats.jsonnet'
    else if $.dataset_name == 'rchestxray' then import '../constants/datasets/chestxray.jsonnet'
    else if $.dataset_name == 'Animals' then import '../constants/datasets/Animals.jsonnet'
    else if $.dataset_name == 'Boats' then import '../constants/datasets/Boats.jsonnet'
    else if $.dataset_name == 'Chars74K' then import '../constants/datasets/Chars74K.jsonnet'
    else if $.dataset_name == 'Dogs' then import '../constants/datasets/Dogs.jsonnet'
    else if $.dataset_name == 'EMNIST_Balanced' then import '../constants/datasets/EMNIST_Balanced.jsonnet'
    else if $.dataset_name == 'EMNIST_Digits' then import '../constants/datasets/EMNIST_Digits.jsonnet'
    else if $.dataset_name == 'EMNIST_Letters' then import '../constants/datasets/EMNIST_Letters.jsonnet'
    else if $.dataset_name == 'EMNIST_MNIST' then import '../constants/datasets/EMNIST_MNIST.jsonnet'
    else if $.dataset_name == 'FashionMNIST' then import '../constants/datasets/FashionMNIST.jsonnet'
    else if $.dataset_name == 'Fruits360' then import '../constants/datasets/Fruits360.jsonnet'
    else if $.dataset_name == 'KMNIST' then import '../constants/datasets/KMNIST.jsonnet'
    else if $.dataset_name == 'cactus' then import '../constants/datasets/cactus.jsonnet'
    else if $.dataset_name == 'kannada' then import '../constants/datasets/kannada.jsonnet'
    else if $.dataset_name == 'kannada_dig' then import '../constants/datasets/kannada_dig.jsonnet'
    else if $.dataset_name == 'VOC07+12_DET' then import '../constants/datasets/VOC07+12_DET.jsonnet'
    else if $.dataset_name == 'cellular' then import '../constants/datasets/cellular.jsonnet'
    else if $.dataset_name == 'underwater' then import '../constants/datasets/underwater.jsonnet'
    else if $.dataset_name == 'aliproducts' then import '../constants/datasets/aliproducts.jsonnet'
    else if $.dataset_name == 'satellite_maps' then import '../constants/datasets/satellite_maps.jsonnet'
    else {},

    // basic type node generator function
    bool(id, en, cn='', def=false, ddd=true, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'bool',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    int(id, en, cn='', def=0, ddd=true, tips='', min=-999666, max=-999666, width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'int',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if min != -999666 then 'min']: min,
        [if max != -999666 then 'max']: max,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    float(id, en, cn='', def=0, ddd=true, tips='', min=-999666, max=-999666, width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'float',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if min != -999666 then 'min']: min,
        [if max != -999666 then 'max']: max,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    string(id, en, cn='', def='', ddd=true, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'string',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    text(id, en, cn='', def='', ddd=true, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'text',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    image(id, en, cn='', def='', ddd=true, tips='', width=-1, height=-1):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'image',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        readonly: true,
    },

    sampleimage(id, en, cn='', def='', ddd=true, tips='', width=-1, height=-1):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'image',
        default: if ddd then $.dataset_root + $.get_default_value(id, def) else $.dataset_root + def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        readonly: true,
    },


    intarray(id, en, cn='', def=[], ddd=true, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'int-array',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    floatarray(id, en, cn='', def=[], ddd=true, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'float-array',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    stringarray(id, en, cn='', def='', ddd=true, tips='', width=-1, height=-1, readonly=false):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'string-array',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    booltrigger(id, en, cn='', def=false, ddd=true, tips='', width=-1, height=-1, readonly=false, trigger=[]):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'bool-trigger',
        objs: [
            {
                name: { en: 'Enable', cn: self.en },
                value: true,
                trigger: {
                    type: '_ignore_',
                    objs: trigger,
                },
            },
            {
                name: { en: 'Disable', cn: self.en },
                value: false,
                trigger: {},
            },
        ],
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },

    stringenum(id, en, cn='', def='', ddd=true, tips='', width=-1, height=-1, readonly=false, enums=[]):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'string-enum',
        objs: enums,
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        [if readonly then 'readonly']: readonly,
    },
}
