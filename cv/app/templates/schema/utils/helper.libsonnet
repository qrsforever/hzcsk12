// @file helper.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 13:31

{
    version:: '0.0.1b',
    task:: std.extVar('task'),
    dataset_name:: std.extVar('dataset_name'),
    dataset_path:: std.extVar('dataset_root') + '/' + $.dataset_name,
    checkpt_root:: std.extVar('checkpt_root'),
    pretrained_path:: std.extVar('pretrained_path'),

    // usage: get_value(confg, 'a.b.c', 100)
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
        [if $.dataset_name == 'mnist' then 'mnist']: (import
                                                          '../constants/datasets/mnist.jsonnet').get($.dataset_path,
                                                                                                     $.checkpt_root),
        [if $.dataset_name == 'cifar10' then 'cifar10']: (import
                                                              '../constants/datasets/cifar10.jsonnet').get($.dataset_path,
                                                                                                           $.checkpt_root),
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

    notiml():: {
        _id_: 'notiml',
        name: { en: 'todo', cn: self.en },
        type: 'text',
        default: 'Not Impl Yet',
    },
}