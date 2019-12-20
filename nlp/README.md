# 自然语言处理

# 目录

   nlp 
    ├── app      # hzcsai对allennlp框架的补充修改
    └── allennlp # 源码: https://github.com/allenai/allennlp/

# allennlp修改原则

修改之处必须加标记, 如:QRS, 方便搜索

## 标记登记

QRS

# allennlp修改记录

| 文件 | 原因 |
|:----:|:----:|
| `commands/__init__.py` | 捕获异常 |
| `training/util.py` | 收集metrics |
| `training/trainer.py` | 收集metrics |
