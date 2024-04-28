import models, torch


class Server(object):
    # 将配置信息拷贝到服务端中
    def __init__(self, conf, eval_dataset):

        self.conf = conf

        self.global_model = models.get_model(self.conf["model_name"])

        # batch_size:每个批次加载的样本数量,shuffle=True表示打乱样本
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    # 模型聚合，即将多个客户端上传的模型参数进行合并，更新全局模型。
    def model_aggregate(self, weight_accumulator):

        # 遍历全局模型的状态字典 self.global_model.state_dict().items()，其中 name 是参数的名称，data 是参数的数据。
        for name, data in self.global_model.state_dict().items():

            # update_per_layer:将 weight_accumulator 中对应参数的累加值乘以配置中的 "lambda" 值得到的。
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]

            # 如果全局模型中的参数数据类型与更新量的数据类型不一致，那么将更新量转换为 torch.int64 类型后再进行累加。
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    # 用来评估模型
    def model_eval(self):
        # 将模型设置为评估模式
        self.global_model.eval()

        # 累计所有批次的损失
        total_loss = 0.0

        # 计数预测正确的样本数
        correct = 0

        # 记录数据集中样本的总数。
        dataset_size = 0

        # 通过 enumerate(self.eval_loader) 遍历数据加载器中的每个批次。data 和 target 分别是输入数据和标签。
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            # 表示对输入数据进行前向传播，得到模型的输出
            output = self.global_model(data)

            # 表示对输入数据进行前向传播，得到模型的输出
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()

            # 获取概率最大的预测类别。
            pred = output.data.max(1)[1]

            # 计算预测正确的样本数，并累加到 correct
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        # 计算总的准确率 acc 和平均损失 total_l，并将它们作为函数的返回值
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l
