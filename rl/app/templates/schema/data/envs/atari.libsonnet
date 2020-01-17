// @file atari.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-17 14:58

local _Utils = import '../../utils/helper.libsonnet';

{
    get(jid): {
        type: 'H',
        objs: [
            _Utils.string(jid + '.game', 'Game', def=_Utils.dataset_name, readonly=true),
            _Utils.int(jid + '.frame_skip', 'Frame Skip', def=4, min=1),
            _Utils.int(jid + '.num_img_obs', 'Observe Number', def=4, min=1),
            _Utils.bool(jid + '.clip_reward', 'Clip Reward', def=true),
            _Utils.bool(jid + '.episodic_lives', 'Episodic Lives', def=if jid == 'env' then true else false),
            _Utils.int(jid + '.max_start_noops', 'Max Noops', def=30),
            _Utils.int(jid + '.repeat_action_probability', 'Repeat Prob.', def=0.0),
            _Utils.int(jid + '.horizon', 'Horizon', def=27000),
        ],
    },
}
