// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-08 12:15

local _Utils = import '../../utils/helper.libsonnet';

local _TypeSelector = {
    cls: [
        {
            name: { en: 'Base', cn: self.en },
            value: 'base_model',
        },
        {
            name: { en: 'Cls', cn: self.en },
            value: 'cls_model',
        },
        {
            name: { en: 'Distill', cn: self.en },
            value: 'distill_model',
        },
        {
            name: { en: 'Custom', cn: self.en },
            value: 'custom_model',
        },
    ],
};

{
    get():: {
        type: 'H',
        objs: [
            {
                _id_: 'network.model_name',
                name: { en: 'Type', cn: self.en },
                type: 'string-enum',
                objs: _TypeSelector[_Utils.task],
                default: _Utils.get_default_value(self._id_, self.objs[0].value),
            },
            (import '../backbone/__init__.jsonnet').get('network'),
        ],
    },
}
