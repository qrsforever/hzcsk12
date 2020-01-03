// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 21:37

local _Utils = import '../../utils/helper.libsonnet';

local _MODELS = {
    basic(jid):: {
        name: { en: 'Basic Classifier', cn: self.en },
        value: 'basic_classifier',
        trigger: (import 'basic_classifier.libsonnet').get(jid),
    },
    bcn(jid):: {
        name: { en: 'BCN', cn: self.en },
        value: 'bcn',
        trigger: (import 'bcn.libsonnet').get(jid),
    },
};

local _SELECTS = {
    sst: {
        get(jid):: [
            _MODELS.basic(jid),
            _MODELS.bcn(jid),
        ],
    },
};

{
    get(jid):: {
        _id_: jid + '.type',
        name: { en: 'Type', cn: self.en },
        type: 'string-enum-trigger',
        objs: _SELECTS[_Utils.dataset_name].get(jid),
        default: self.objs[0].value,
    },
}
