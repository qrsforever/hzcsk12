// @file random_perm.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2019-12-11 23:50

local lib = import 'common.libsonnet';

{
    get(prefix): {
        local this = self,
        _id_:: prefix + '.aug_trans.random_perm',
        name: { en: 'Permutation', cn: self.en },
        value: 'random_perm',
        trigger: {
            name: this.name,
            type: 'object',
            objs: [
                lib.radio(this._id_ + '.ratio'),
            ],
        },
    },
}
