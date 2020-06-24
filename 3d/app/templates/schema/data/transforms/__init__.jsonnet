// @file __init__.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-24 11:03

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid):: {
        type: 'V',
        objs: [
            {
                type: 'H',
                objs: [
                    _Utils.intarray(jid + '.output_size',
                                    'Output Size',
                                    def=[28, 28],
                                    tips='the output size of transform'),
                    _Utils.intarray(jid + '.normalize.mean',
                                    'Normal Mean',
                                    def=[0.5, 0.5, 0.5]),
                    _Utils.intarray(jid + '.normalize.std',
                                    'Normal Std',
                                    def=[0.5, 0.5, 0.5]),
                    _Utils.bool(jid + '.shuffle', 'Shuffle', def=false, tips='shuffle the transform'),
                ],
            },
            {
                name: { en: 'Compose', cn: self.en },
                type: 'navigation',
                objs: [
                    {
                        name: { en: 'Resize', cn: self.en },
                        type: '_ignore_',
                        objs: [
                            _Utils.booltrigger(
                                '_k12.' + jid + '.compose.resize',
                                'Enable',
                                def=false,
                                trigger=(import 'type/resize.libsonnet').get(jid + '.compose.resize')
                            ),
                        ],
                    },
                    {
                        name: { en: 'CenterCrop', cn: self.en },
                        type: '_ignore_',
                        objs: [
                            _Utils.booltrigger(
                                '_k12.' + jid + '.compose.center_crop',
                                'Enable',
                                def=false,
                                trigger=(import 'type/center_crop.libsonnet').get(jid + '.compose.center_crop')
                            ),
                        ],
                    },
                    {
                        name: { en: 'RandomHFlip', cn: self.en },
                        type: '_ignore_',
                        objs: [
                            _Utils.booltrigger(
                                '_k12.' + jid + '.compose.random_horizontal_flip',
                                'Enable',
                                def=false,
                                trigger=(import 'type/random_horizontal_flip.libsonnet').get(jid + '.compose.random_horizontal_flip')
                            ),
                        ],
                    },
                    {
                        name: { en: 'ColorJitter', cn: self.en },
                        type: '_ignore_',
                        objs: [
                            _Utils.booltrigger(
                                '_k12.' + jid + '.compose.color_jitter',
                                'Enable',
                                def=false,
                                trigger=(import 'type/color_jitter.libsonnet').get(jid + '.compose.color_jitter')
                            ),
                        ],
                    },
                ],
            },
        ],
    },
}
