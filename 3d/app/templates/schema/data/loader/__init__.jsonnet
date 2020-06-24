// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-23 22:35

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'V',
        objs: [
            {
                _id_: jid + '.type',
                name: { en: 'Loader Type', cn: self.en },
                type: 'string-enum',
                tips: 'load dataset type',
                objs: [
                    {
                        name: { en: 'jsonfile', cn: self.en },
                        value: 'jsonfile',
                    },
                    {
                        name: { en: 'listdir', cn: self.en },
                        value: 'listdir',
                    },
                ],
                default: 'listdir',
            },
            {
                name: { en: 'Loader Parameters', cn: self.en },
                type: 'H',
                objs: [
                    _Utils.stringenum(jid + '.args.batch_size',
                                      'Batch Size',
                                      def=32,
                                      tips='samples count per batch to load on training',
                                      enums=[
                                          { name: { en: '16', cn: self.en }, value: 16 },
                                          { name: { en: '32', cn: self.en }, value: 32 },
                                          { name: { en: '64', cn: self.en }, value: 64 },
                                          { name: { en: '128', cn: self.en }, value: 128 },
                                          { name: { en: '256', cn: self.en }, value: 256 },
                                      ]),
                    _Utils.int(jid + '.args.num_workers',
                               'Jobs',
                               def=2,
                               min=1,
                               max=_Utils.num_cpu,
                               tips='the numbers of subprocesses for loading dataset'),

                    _Utils.bool(jid + '.args.shuffle',
                                'Shuffle',
                                def=true),
                    _Utils.bool(jid + '.args.drop_last',
                                'Drop Last',
                                def=true),
                    _Utils.bool(jid + '.args.pin_memory',
                                'Pin Mem',
                                def=true),
                ],
            },
        ],
    },
}
