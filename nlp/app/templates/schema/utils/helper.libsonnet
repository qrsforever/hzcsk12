// @file helper.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-30 12:14

local _network_maps = {
    basic_classifier: {
        name: { en: 'Basic Classifier', cn: self.en },
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
    network_name:: if std.objectHas(_network_maps, $.network) then _network_maps[$.network].name else 'unkown',
    dataset_name:: std.extVar('dataset_name'),
    // notebook_url:: 'http://' + $.net_ip + ':8118/notebooks/nlp/tasks',

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
    datasets: {
        [if $.dataset_name == 'sst' then 'sst']: (import '../constants/datasets/sst.jsonnet'),
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
