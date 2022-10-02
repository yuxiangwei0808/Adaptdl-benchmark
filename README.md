Bench 移植中遇到的问题
===================

1. Imagenet 没有 criterion，是用 Model 自带的 loss，这个怎么优化。

outputs = model(**batch)

2. Initialize 函数怎么 general 地传东西给 step 函数。

eg: ncf 的 lr_scheduler 每个 epoch 需要调用一次，但是 initilize 初始化的东西只能直接传递给 step，不能传递给 epoch 的函数。

```python
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [4, 7, 10], gamma=0.2)
```