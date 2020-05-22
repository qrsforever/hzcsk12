// @file init.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-02 21:37

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: [
        {
            type: 'H',
            objs: [
                {
                    _id_: jid + '.type',
                    name: { en: 'Network', cn: self.en },
                    type: 'string-enum',
                    objs: [
                        {
                            name: _Utils.network_name,
                            value: _Utils.network,
                        },
                    ],
                    default: self.objs[0].value,
                    readonly: true,
                },
                _Utils.bool('_k12.model.resume', 'Resume', def=false),
            ],
        },
    ] + if _Utils.network == 'basic_classifier'
    then
        (import 'basic_classifier.libsonnet').get(jid)
    else if _Utils.network == 'crf_tagger'
    then
        (import 'crf_tagger.libsonnet').get(jid)
    else [],
}
