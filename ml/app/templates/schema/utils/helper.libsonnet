// @file helper.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-02-11 23:18

local _network_maps = {
    svc: {
        method: 'sklearn_wrapper',
        name: { en: 'svc', cn: self.en },
    },
    svr: {
        method: 'sklearn_wrapper',
        name: { en: 'svr', cn: self.en },
    },
    knn: {
        method: 'sklearn_wrapper',
        name: { en: 'knn', cn: self.en },
    },
    kmeans: {
        method: 'sklearn_wrapper',
        name: { en: 'kmeans', cn: self.en },
    },
    gaussian_nb: {
        method: 'sklearn_wrapper',
        name: { en: 'Gaussian NB', cn: self.en },
    },
    decision_tree: {
        method: 'sklearn_wrapper',
        name: { en: 'decision tree', cn: self.en },
    },
    random_forest: {
        method: 'sklearn_wrapper',
        name: { en: 'random forest', cn: self.en },
    },
    adaboost: {
        method: 'sklearn_wrapper',
        name: { en: 'adaptive boosting', cn: self.en },
    },
    logistic: {
        method: 'sklearn_wrapper',
        name: { en: 'logistic', cn: self.en },
    },
    gradient_boosting: {
        method: 'sklearn_wrapper',
        name: { en: 'gradient boosting', cn: self.en },
    },
    xgboost: {
        method: 'xgboost_wrapper',
        name: { en: 'xgboost', cn: self.en },
    },
};

{
    levelid:: std.extVar('levelid'),
    debug:: std.extVar('debug'),
    net_ip:: std.extVar('net_ip'),
    num_cpu:: std.extVar('num_cpu'),
    num_gpu:: std.extVar('num_gpu'),
    task:: std.extVar('task'),
    network:: std.extVar('network'),
    method:: if std.objectHas(_network_maps, $.network) then _network_maps[$.network].method else 'unkown',
    network_name:: if std.objectHas(_network_maps, $.network) then _network_maps[$.network].name else 'unkown',
    dataset_name:: std.extVar('dataset_name'),
    notebook_url:: 'http://' + $.net_ip + ':8118/notebooks/ml/tasks',

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
        [if $.dataset_name == 'iris' then 'iris']: import '../constants/datasets/iris.jsonnet',
        [if $.dataset_name == 'digits' then 'digits']: import '../constants/datasets/digits.jsonnet',
        [if $.dataset_name == 'diabetes' then 'diabetes']: import '../constants/datasets/diabetes.jsonnet',
        [if $.dataset_name == 'wine' then 'wine']: import '../constants/datasets/wine.jsonnet',
        [if $.dataset_name == 'boston' then 'boston']: import '../constants/datasets/boston.jsonnet',
        [if $.dataset_name == 'linnerud' then 'linnerud']: import '../constants/datasets/linnerud.jsonnet',
        [if $.dataset_name == 'breast_cancer' then 'breast_cancer']: import '../constants/datasets/breast_cancer.jsonnet',

        [if $.dataset_name == 'sf-crime' then 'sf-crime']: import '../constants/datasets/sf-crime.jsonnet',
        [if $.dataset_name == 'titanic' then 'titanic']: import '../constants/datasets/titanic.jsonnet',
        [if $.dataset_name == 'house-prices' then 'house-prices']: import '../constants/datasets/house-prices.jsonnet',
    },

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
