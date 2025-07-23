# Chapter0-PyTorch入门

> 参考: https://www.runoob.com/pytorch/pytorch-tutorial.html , 遇到不会的函数可以查看 `PyTorch` 官方文档: https://docs.pytorch.org/docs/stable/data.html

`PyTorch` 作为常用的深度学习框架, 提供了很多好用的工具, 但是我认为只需要掌握如下几类工具:

1. `torch`:  **提供各种与 `torch.tensor` 相关的操作以及各种函数(比如 `relu, log` 等), 以及各种 `torch.tensor` 的操作函数, 同时注意 `torch.Tensor` 中的属性以及相关的操作函数**    [`torch`](https://docs.pytorch.org/docs/stable/torch.html#)
2. `torch.nn`:  **提供 `nn.Module` 类, 预定义层(`nn.Linear` 等), 容器类(`nn.Sequential`) 等, 损失函数(`nn.CrossEntropyLoss`) 等, 以及一些使用函数接口(在 `torch.nn.functional`) 中,**  [`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html)  [`torch.nn.functional`](https://docs.pytorch.org/docs/stable/nn.functional.html)
3. `torch.utils.data`: **提供数据集加载等工具, 提供 `Dataset, DataLoader, ConcatDataset` 等**   [`torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html)

4. `torchvision`: **提供数据集, 以及与数据集相关的转换工具**, 参考: [`torchvision`](https://docs.pytorch.org/vision/stable/) 