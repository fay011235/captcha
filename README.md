# captcha
验证码识别--迁移学习
两种方法：冻结法+微调法
冻结法是直接将网络层参数冻结，而只训练最后的全连接层。
微调法是微调网络参数，而重新初始化最后的全连接层。
在训练样本量较多时，样本和原模型训练样本存在较大差别时，可以使用微调法。而在样本量较少时或样本和原模型训练样本比较相似时，可以使用冻结法。
由于原模型参数较多，使用微调法消耗的内存也会较多。
