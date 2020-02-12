// @file assemble.jsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-06 11:44

local _Utils = import '../utils/helper.libsonnet';

[
    {
        type: 'H',
        objs: [
            _Utils.string('task', 'Task', def=_Utils.task, readonly=true, tips='task type'),
            _Utils.string('method', 'Method', def=_Utils.method, ddd=true, readonly=true),
            _Utils.bool('data.include_val', 'Include Val', def=false),
        ],
    },
    {
        type: 'H',
        objs: [
            _Utils.string('dataset', 'Loader', def='default', ddd=true, readonly=true),
            _Utils.int('data.workers', 'Workers', min=1, max=8, def=4),
            _Utils.bool('data.drop_last', 'Drop Last', def=false),
        ],
    },
    {
        type: 'H',
        objs: [
            {
                _id_: 'data.image_tool',
                name: { en: 'Image Tool', cn: self.en },
                type: 'string-enum',
                objs: [
                    {
                        name: { en: 'PIL', cn: self.en },
                        value: 'pil',
                    },
                    {
                        name: { en: 'CV2', cn: self.en },
                        value: 'cv2',
                    },
                ],
                default: _Utils.get_default_value(self._id_, self.objs[0].value),
            },
            {
                _id_: 'data.input_mode',
                name: { en: 'Input Mode', cn: self.en },
                type: 'string-enum',
                objs: [
                    {
                        name: { en: 'RGB', cn: self.en },
                        value: 'RGB',
                    },
                    {
                        name: { en: 'BGR', cn: self.en },
                        value: 'BGR',
                    },
                    {
                        name: { en: 'GRAY', cn: self.en },
                        value: 'GRAY',
                    },
                ],
                default: _Utils.get_default_value(self._id_, self.objs[0].value),
            },
            (if _Utils.task == 'det' || _Utils.task == 'ins' then
                 _Utils.bool('data.keep_difficult', 'Keep Difficult', def=false) else {}),
        ],
    },
    {
        type: 'accordion',
        objs: [
            {
                name: { en: 'Iterator', cn: self.en },
                type: '_ignore_',
                objs: (import 'iterator/__init__.jsonnet').get(),
            },
            {
                name: { en: 'Transform', cn: self.en },
                type: '_ignore_',
                objs: (import 'transform/__init__.jsonnet').get(),
            },
            {
                name: { en: 'Details', cn: self.en },
                type: '_ignore_',
                objs: (import 'details/__init__.jsonnet').get(),
            },
        ],
    },
]
