// @file color_jitter.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-06-24 18:38

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        {
            type: 'H',
            objs: [
                _Utils.float(jid + '.args.brightness', 'Brightness', def=0.5, tips='How much to jitter brightness'),
                _Utils.float(jid + '.args.contrast', 'Contrast', def=0.5, tips='How much to jitter contrast'),
                _Utils.float(jid + '.args.saturation', 'Saturation', def=0.5, tips='How much to jitter saturation'),
                _Utils.float(jid + '.args.hue', 'Hue', def=0.5, tips='How much to jitter hue'),
            ],
        },
        _Utils.checkboxphase(jid),
    ],
}
