https://rlpyt.readthedocs.io/
https://github.com/astooke/rlpyt     

# rlpyt最新更新

```
commit d797dd8835a91a7b2902563a097c83ed7c8d3e92
Date:   Mon Jan 27 13:30:36 2020 -0800
```

```
commit a865f6712a049f9fd26500e924114e9582a6a5c2
Date:   Mon Apr 6 18:20:33 2020 -0700
```

# rlpyt修改记录

| 文件 | 原因 |
|:----|:----|
|`rlpyt/rlpyt/utils/logging/logger.py` | hook log |
|`rlpyt/rlpyt/samplers/parallel/worker.py` | 解决work进程异常导致主进程死等 |
|`rlpyt/rlpyt/runners/minibatch_rl.py` | fix bug: get `min_itr_learn` is 0 |
|`rlpyt/rlpyt/algos/dqn/dqn.py` | fix bug: #152 |

# ISSUES

- [Exception in logger with Pytorch >= 1.4](https://github.com/astooke/rlpyt/issues/152)

# TODO

| 文件 | 函数 | 问题 |
|:----|:----|:----|
|`rlpyt/rlpyt/samplers/parallel/base.py` | `_assemble_workers_kwargs` | `samples_np`参数函数内外会变化, 不知原因, 放弃解决 |
