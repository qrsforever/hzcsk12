// @file random_border.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 20:00

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.radio', 'radio', min=0, max=1.0, def=0.5, tips='the radio of using the method'),
        _Utils.intarray(jid + '.pad',
                        'pad',
                        def=[0, 0, 0, 0],
                        tips='value like [int, int, int, int] meaning [left, top, right, bottom] of padding board, every element >= 0'),
        _Utils.intarray(jid + '.mean', 'mean', def=[104, 117, 124], tips='padding fill value, like [r, g, b], every element between 0 and 255'),
    ] + if _Utils.task == 'det'
    then _Utils.bool(jid + '.allow_outsize_center', 'outsize center', def=true) else [],
}
