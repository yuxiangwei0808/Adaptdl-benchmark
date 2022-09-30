Bench 移植中遇到的问题
===================

1. Imagenet 没有 criterion，是用 Model 自带的 loss，这个怎么优化。

outputs = model(**batch)

2. Initialize 函数怎么 general 地传东西给 step 函数。
