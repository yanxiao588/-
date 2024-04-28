import models, torch, copy


# 客户端
class Client(object):
    # 将配置信息拷贝到客户端
    def __init__(self, conf, model, train_dataset, id=-1):

        self.conf = conf

        self.local_model = models.get_model(self.conf["model_name"])

        self.client_id = id

        self.train_dataset = train_dataset

        # 创建一个列表all_range，包含从0到训练数据集大小减1的所有整数。这些整数可以看作是训练数据集中每个样本的索引。
        all_range = list(range(len(self.train_dataset)))

        # 计算每个子集（即每个客户端）应该包含的样本数量。这是通过将训练数据集的总样本数量除以客户端数量得到的。
        data_len = int(len(self.train_dataset) / self.conf['no_models'])

        # 根据客户端IDid，从all_range中切片出对应的样本索引，作为该客户端的训练数据。
        # 例如，如果有10个客户端，那么第0个客户端的训练数据就是all_range的前10%的部分，第1个客户端的训练数据就是all_range的10%到20%的部分，以此类推。
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        # 创建一个数据加载器，它会在每次训练迭代中随机地从客户端的训练数据子集中取出batch_size数量的样本
        # SubsetRandomSampler会在给定的索引列表（即train_indices）中随机选择索引，然后从数据集中取出对应的样本。这样，每个客户端就只会看到属于自己的数据子集。
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))

    # 在客户端上进行本地训练
    def local_train(self, model):

        # 遍历全局模型的所有参数。model.state_dict().items()返回一个包含模型所有参数的字典，其中name是参数的名称，param是参数的值。
        # 将全局模型的参数复制到本地模型。param.clone()创建参数的一个副本，copy_方法将这个副本的值复制到本地模型的对应参数。
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # 创建了一个随机梯度下降（SGD）优化器，用于更新本地模型的参数。
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])

        # 将本地模型设置为训练模式。
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):

            # 遍历训练数据加载器中的所有批次。每个批次包含一组数据和对应的目标。
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                # 检查是否有可用的CUDA设备。如果有，就将数据和目标移动到CUDA设备上，以便在GPU上进行计算。
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                # 将优化器中所有参数的梯度清零。这是因为PyTorch默认会累积梯度，所以在每次训练迭代开始时需要清零梯度。
                optimizer.zero_grad()

                # 数据传递给本地模型，得到模型的输出。
                output = self.local_model(data)

                # 计算模型输出和目标之间的交叉熵损失。
                loss = torch.nn.functional.cross_entropy(output, target)

                # 计算损失相对于模型参数的梯度
                loss.backward()

                optimizer.step()
            print("Epoch %d done." % e)
        diff = dict()

        # 遍历本地模型的所有参数。计算本地模型和全局模型参数的差异，并将差异保存到diff字典中。
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])

        return diff
