# 计算机视觉

# 目录

```
   cv
    ├── app      # hzcsai对torchcv框架的补充修改
    └── torchcv  # 源码: https://github.com/donnyyou/torchcv
```

**torchcv的代码尽量少修改, 方便torchcv更新升级, 通过app植入到torchcv**

# torchcv修改原则

修改之处必须加标记, 如:QRS, 方便搜索

## 标记登记

QRS

# torchcv修改记录

| 文件 | 原因 |
|:----|:----|
| `torchcv/main.py` | 添加hzcs初始函数以及异常捕捉 |
| `torchcv/data/tools/cv2_aug_transforms.py` | det训练RandomResizeCrop异常 |
| `torchcv/tools/util/logger.py` | 截获日志, 捕捉错误 |
| `torchcv/model/cls/nets/base_model.py` | 运行错误fix bug |
| `torchcv/model/cls/loss/mixup_ce_loss.py` | fix bug |
| `torchcv/model/cls/loss/mixup_soft_ce_loss.py` | fix bug |
