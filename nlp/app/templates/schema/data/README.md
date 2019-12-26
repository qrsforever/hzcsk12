---

title: Data页标签配置

date: 2019-12-26 17:01
tags: [draft, k12nlp]
categories: [k12ai]

---

```
                                  task
                                    |                       ccgbank |
                                    |                           sst |
                                    v                          quac |
             constants <-------- datasets                       srl |
                                    |             language_modeling |
                                    |                         coref |
                                    v                               |
                                 readers ---------------------------+
                                    |
                                    |
                                    v
                     +--------------+-------------+
                     |                            |
                     |                            |
                     v                            v
                token_indexers                tokenizer
                     |                            |
                     |                            |
   +----------+------+--------+-----------+       +-----> word
   |          |               |           |       |
   v          v               v           v       |
pos_tag   single_id   token_characters  ner_tag   +-----> character
```
