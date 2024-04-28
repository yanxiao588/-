import argparse, json
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets

# 检查当前脚本是否作为主程序运行。如果是，那么就执行后面的代码。
if __name__ == '__main__':

    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(description='Federated Learning')

    # 向解析器中添加一个命令行参数。这个参数的名称是'--conf'，短名称是'-c'，目标变量是'conf'。
    parser.add_argument('-c', '--conf', dest='conf')

    # 解析命令行参数，并将结果保存到args中。
    args = parser.parse_args()

    # 打开配置文件进行读取。
    with open(args.conf, 'r') as f:
        conf = json.load(f)  # 从配置文件中加载JSON数据，并将结果保存到conf中。

    # 加载训练数据集和评估数据集。
    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []

    # 创建多个客户端对象，并将它们添加到客户端列表中。
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    print("\n\n")
    # 开始全局训练循环。
    for e in range(conf["global_epochs"]):

        # 在每轮全局训练中，随机选择k个客户端进行训练。
        candidates = random.sample(clients, conf["k"])

        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        # 对于每个选中的客户端，执行本地训练，并计算模型参数的更新。
        for c in candidates:
            diff = c.local_train(server.global_model)

            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        # 服务器聚合所有客户端的模型更新。
        server.model_aggregate(weight_accumulator)

        # 服务器评估全局模型的性能。
        acc, loss = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
