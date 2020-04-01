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
    custom_base: {
        method: 'image_classifier',
        name: { en: 'base', cn: self.en },
    },
    vgg16_ssd300: {
        method: 'single_shot_detector',
        name: { en: 'ssd300', cn: self.en },
    },
    vgg16_ssd512: {
        method: 'single_shot_detector',
        name: { en: 'ssd512', cn: self.en },
    },
    custom_ssd300: {
        method: 'single_shot_detector',
        name: { en: 'ssd300', cn: self.en },
    },
    custom_ssd512: {
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
    debug:: std.extVar('debug'),
    net_ip:: std.extVar('net_ip'),
    task:: std.extVar('task'),
    num_cpu:: std.extVar('num_cpu'),
    num_gpu:: std.extVar('num_gpu'),
    network:: std.extVar('network'),
    method:: if std.objectHas(_network_maps, $.network) then _network_maps[$.network].method else 'unkown',
    network_name:: if std.objectHas(_network_maps, $.network) then _network_maps[$.network].name else 'unkown',
    dataset_name:: std.extVar('dataset_name'),
    notebook_url:: 'http://' + $.net_ip + ':8118/notebooks/cv/tasks/' +
                   $.task + '_' + $.network + '_' + $.dataset_name + '.ipynb',
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
        if std.objectHas($.datasets, $.dataset_name)
        then
            $.get_value($.datasets[$.dataset_name], keystr, def)
        else
            def,

    // default dataset, can set default value
    datasets:: {
        [if $.dataset_name == 'mnist' then 'mnist']: import '../constants/datasets/mnist.jsonnet',
        [if $.dataset_name == 'cifar10' then 'cifar10']: import '../constants/datasets/cifar10.jsonnet',
        [if $.dataset_name == 'Animals' then 'Animals']: import '../constants/datasets/Animals.jsonnet',
        [if $.dataset_name == 'Boats' then 'Boats']: import '../constants/datasets/Boats.jsonnet',
        [if $.dataset_name == 'Chars74K' then 'Chars74K']: import '../constants/datasets/Chars74K.jsonnet',
        [if $.dataset_name == 'dogsVsCats' then 'dogsVsCats']: import '../constants/datasets/dogsVsCats.jsonnet',
        [if $.dataset_name == 'Dogs' then 'Dogs']: import '../constants/datasets/Dogs.jsonnet',
        [if $.dataset_name == 'EMNIST_Balanced' then 'EMNIST_Balanced']: import '../constants/datasets/EMNIST_Balanced.jsonnet',
        [if $.dataset_name == 'EMNIST_Digits' then 'EMNIST_Digits']: import '../constants/datasets/EMNIST_Digits.jsonnet',
        [if $.dataset_name == 'EMNIST_Letters' then 'EMNIST_Letters']: import '../constants/datasets/EMNIST_Letters.jsonnet',
        [if $.dataset_name == 'EMNIST_MNIST' then 'EMNIST_MNIST']: import '../constants/datasets/EMNIST_MNIST.jsonnet',
        [if $.dataset_name == 'FashionMNIST' then 'FashionMNIST']: import '../constants/datasets/FashionMNIST.jsonnet',
        [if $.dataset_name == 'Fruits360' then 'Fruits360']: import '../constants/datasets/Fruits360.jsonnet',
        [if $.dataset_name == 'KMNIST' then 'KMNIST']: import '../constants/datasets/KMNIST.jsonnet',
        [if $.dataset_name == 'cactus' then 'cactus']: import '../constants/datasets/cactus.jsonnet',
        [if $.dataset_name == 'kannada' then 'kannada']: import '../constants/datasets/kannada.jsonnet',
        [if $.dataset_name == 'kannada_dig' then 'kannada_dig']: import '../constants/datasets/kannada_dig.jsonnet',
        [if $.dataset_name == 'VOC07+12_DET' then 'VOC07+12_DET']: import '../constants/datasets/VOC07+12_DET.jsonnet',
        [if $.dataset_name == 'cellular' then 'cellular']: import '../constants/datasets/cellular.jsonnet',
        [if $.dataset_name == 'underwater' then 'underwater']: import '../constants/datasets/underwater.jsonnet',
        [if $.dataset_name == 'aliproducts' then 'aliproducts']: import '../constants/datasets/aliproducts.jsonnet',
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

    int(id, en, cn='', def=0, ddd=false, tips='', min=-999666, max=-999666, width=-1, height=-1, readonly=false):: {
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

    float(id, en, cn='', def=0, ddd=false, tips='', min=-999666, max=-999666, width=-1, height=-1, readonly=false):: {
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

    image(id, en, cn='', def='', ddd=false, tips='', width=-1, height=-1):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'image',
        default: if ddd then $.get_default_value(id, def) else def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        readonly: true,
    },

    sampleimage(id, en, cn='', def='', ddd=false, tips='', width=-1, height=-1):: {
        _id_: id,
        name: { en: en, cn: if std.length(cn) == 0 then self.en else cn },
        type: 'image',
        default: if ddd then $.dataset_root + $.get_default_value(id, def) else $.dataset_root + def,
        [if std.length(tips) > 0 then 'tips']: tips,
        [if width > 0 then 'width']: width,
        [if height > 0 then 'height']: height,
        readonly: true,
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

    booltrigger(id, en, cn='', def=false, ddd=false, tips='', width=-1, height=-1, readonly=false, trigger=[]):: {
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

    stringenum(id, en, cn='', def='', ddd=false, tips='', width=-1, height=-1, readonly=false, enums=[]):: {
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
