# 自然语言处理

# 目录

```
   nlp 
    ├── app      # hzcsai对allennlp框架的补充修改
    └── allennlp # 源码: https://github.com/allenai/allennlp/
```

# allennlp修改原则

修改之处必须加标记, 如:QRS, 方便搜索

## 标记登记

QRS

# allennlp修改记录

| 文件 | 原因 |
|:----|:----|
| `allennlp/allennlp/commands/__init__.py` | 捕获异常 |
| `allennlp/allennlp/training/util.py` | 收集metrics |
| `allennlp/allennlp/training/trainer.py` | 收集metrics |


# TODO

版本有问题: `err_type': 'KeyError', 'err_text': "'num_tokens'`
```
commit bae0c55e8c447811d9fa13f1c48f3e2576ab0dcc
Date:   Tue Feb 4 18:43:32 2020 -0800
```
