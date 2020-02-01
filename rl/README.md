https://rlpyt.readthedocs.io/
https://github.com/astooke/rlpyt     

# rlpyt 修改记录

| 文件 | 原因 |
|:----|:----|
|`rlpyt/rlpyt/utils/logging/logger.py` | hook log |
|`rlpyt/rlpyt/samplers/parallel/worker.py` | 解决work进程异常导致主进程死等 |


# TODO

| 文件 | 函数 | 问题 |
|:----|:----|:----|
|`rlpyt/rlpyt/samplers/parallel/base.py` | `_assemble_workers_kwargs` | `samples_np`参数函数内外会变化, 不知原因, 放弃解决 |
