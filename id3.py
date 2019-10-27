import numpy as np
import pandas as pd
import json


class LoadDataSet(object):
    def load_dataSet(self):
        """数据文件下载地址：https://files.cnblogs.com/files/kuaizifeng/ID3data.txt.zip"""
        data = pd.read_csv("ID3data.txt", sep="\t", header=None)
        data.rename(columns={0: "age", 1: "income", 2: "student", 3: "reputation", 4: "purchase"}, inplace=True)
        return data


class TreeHandler(object):
    def __init__(self):
        self.tree = None

    def save(self, tree):
        self.tree = tree
        with open("tree.txt", mode="w", encoding="utf-8") as f:
            tree = json.dumps(tree, indent="  ", ensure_ascii=False)
            f.write(tree)

    def load(self, file):
        with open(file, mode="r", encoding="utf-8") as f:
            tree = f.read()
            self.tree = json.loads(tree)
        return self.tree


class ID3Tree(LoadDataSet, TreeHandler):
    """主要的数据结构是pandas对象"""
    __count = 0

    def __init__(self):
        super().__init__()
        """认定最后一列是标签列"""
        self.dataSet = self.load_dataSet()
        self.gain = {}

    def _entropy(self, dataSet):
        """计算给定数据集的熵"""
        labels = list(dataSet.columns)
        level_count = dataSet[labels[-1]].value_counts().to_dict()  # 统计分类标签不同水平的值
        entropy = 0.0
        for key, value in level_count.items():
            prob = float(value) / dataSet.shape[0]
            entropy += -prob * np.log2(prob)
        return entropy

    def _split_dataSet(self, dataSet, column, level):
        """根据给定的column和其level来获取子数据集"""
        subdata = dataSet[dataSet[column] == level]
        del subdata[column]  # 删除这个划分字段列
        return subdata.reset_index(drop=True)  # 重建索引

    def _best_split(self, dataSet):
        """计算每个分类标签的信息增益"""
        best_info_gain = 0.0  # 求最大信息增益
        best_label = None  # 求最大信息增益对应的标签(字段)
        labels = list(dataSet.columns)[: -1]  # 不包括最后一个靶标签
        init_entropy = self._entropy(dataSet)  # 先求靶标签的香农熵
        for _, label in enumerate(labels):
            # 根据该label(也即column字段)的唯一值(levels)来切割成不同子数据集，并求它们的香农熵
            levels = dataSet[label].unique().tolist()  # 获取该分类标签的不同level
            label_entropy = 0.0  # 用于累加各水平的信息熵；分类标签的信息熵等于该分类标签的各水平信息熵与其概率积的和。
            for level in levels:  # 循环计算不同水平的信息熵
                level_data = dataSet[dataSet[label] == level]  # 获取该水平的数据集
                prob = level_data.shape[0] / dataSet.shape[0]  # 计算该水平的数据集在总数据集的占比
                # 计算香农熵，并更新到label_entropy中
                label_entropy += prob * self._entropy(level_data)  # _entropy用于计算香农熵
            # 计算信息增益
            info_gain = init_entropy - label_entropy  # 代码至此，已经能够循环计算每个分类标签的信息增益
            # 用best_info_gain来取info_gain的最大值，并获取对应的分类标签
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_label = label
            # 这里保存一下每一次计算的信息增益，便于查看和检查错误
            self.gain.setdefault(self.__count, {})  # 建立本次函数调用时的字段，设其value为字典
            self.gain[self.__count][label] = info_gain  # 把本次函数调用时计算的各个标签数据存到字典里
        self.__count += 1
        return best_label

    def _top_amount_level(self, target_list):
        class_count = target_list.value_counts().to_dict()  # 计算靶标签的不同水平的样本量，并转化为字典
        # 字典的items方法可以将键值对转成[(), (), ...]，可以使用列表方法
        sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_class_count[0][0]

    def mktree(self, dataSet):
        """创建决策树"""
        target_list = dataSet.iloc[:, -1]  # target_list 靶标签的那一列数据
        # 程序终止条件一: 靶标签(数据集的最后一列因变量)在该数据集上只有一个水平，返回该水平
        if target_list.unique().shape[0] <= 1:
            return target_list[0]  # ！！！
        # 程序终止条件二: 数据集只剩下把标签这一列数据；返回数量最多的水平
        if dataSet.shape[1] == 1:
            return self._top_amount_level(target_list)
        # 不满足终止条件时，做如下递归处理
        # 1.选择最佳分类标签
        best_label = self._best_split(dataSet)
        # 2.递归计算最佳分类标签的不同水平的子数据集的信息增益
        #   各个子数据集的最佳分类标签的不同水平...
        #   ...
        #   直至递归结束
        best_label_levels = dataSet[best_label].unique().tolist()
        tree = {best_label: {}}  # 生成字典，用于保存树状分类信息；这里不能用self.tree = {}存储
        for level in best_label_levels:
            level_subdata = self._split_dataSet(dataSet, best_label, level)  # 获取该水平的子数据集
            tree[best_label][level] = self.mktree(level_subdata)  # 返回结果
        return tree

    def predict(self, tree, labels, test_sample):
        """
        对单个样本进行分类
        tree: 训练的字典
        labels: 除去最后一列的其它字段
        test_sample: 需要分类的一行记录数据
        """
        firstStr = list(tree.keys())[0]  # tree字典里找到第一个用于分类键值对
        secondDict = tree[firstStr]
        featIndex = labels.index(firstStr)  # 找到第一个建(label)在给定label的索引
        for key in secondDict.keys():
            if test_sample[featIndex] == key:  # 找到test_sample在当前label下的值
                if secondDict[key].__class__.__name__ == "dict":
                    classLabel = self.predict(secondDict[key], labels, test_sample)
                else:
                    classLabel = secondDict[key]
        return classLabel

    def _unit_test(self):
        """用于测试_entropy函数"""
        data = [[1, 1, "yes"],
                [1, 1, "yes"],
                [1, 0, "no"],
                [0, 1, "no"],
                [0, 1, "no"], ]
        data = pd.DataFrame(data=data, columns=["a", "b", "c"])
        # return data # 到此行，用于测试_entropy
        # return self._split_dataSet(data, "a", 1)  # 到此行，用于测试_split_dataSet
        # return self._best_split(data)  # 到此行，用于测试_best_split
        # return self.mktree(self.dataSet)  # 到此行，用于测试主程序mktree
        self.tree = self.mktree(self.dataSet)  # 到此行，用于测试主程序mktree
        labels = ["age", "income", "student", "reputation"]
        test_sample = [0, 1, 0, 0]  # [0, 1, 0, 0, "no"]
        outcome = self.predict(self.tree, labels, test_sample)
        print("The truth class is %s, The ID3Tree outcome is %s." % ("no", outcome))
model = ID3Tree()
model._unit_test()
print(json.dumps(model.gain, indent="  "))  # 可以查看每次递归时的信息熵
print(json.dumps(model.tree, indent="  "))  # 查看树

# The truth class is no, The ID3Tree outcome is no.