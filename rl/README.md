https://rlpyt.readthedocs.io/
https://github.com/astooke/rlpyt     

# rlpyt最新更新

```
commit d797dd8835a91a7b2902563a097c83ed7c8d3e92
Date:   Mon Jan 27 13:30:36 2020 -0800
```

# rlpyt修改记录

| 文件 | 原因 |
|:----|:----|
|`rlpyt/rlpyt/utils/logging/logger.py` | hook log |
|`rlpyt/rlpyt/samplers/parallel/worker.py` | 解决work进程异常导致主进程死等 |
|`rlpyt/rlpyt/envs/gym.py` | 支持离散action游戏如: CartPole |

# TODO

| 文件 | 函数 | 问题 |
|:----|:----|:----|
|`rlpyt/rlpyt/samplers/parallel/base.py` | `_assemble_workers_kwargs` | `samples_np`参数函数内外会变化, 不知原因, 放弃解决 |
