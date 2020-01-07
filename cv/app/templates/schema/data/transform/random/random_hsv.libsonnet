// @file random_hsv.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-07 19:17

local _Utils = import '../../../utils/helper.libsonnet';

{
    get(jid): [
        _Utils.float(jid + '.ratio', 'ratio', def=0.5),
        _Utils.floatarray(jid + '.h_range', 'H range', def=[0.2, 0.8], ddd=true),
        _Utils.floatarray(jid + '.s_range', 'S range', def=[0.2, 0.8], ddd=true),
        _Utils.intarray(jid + '.v_range', 'V range', def=[100, 200], ddd=true),
    ],
}
