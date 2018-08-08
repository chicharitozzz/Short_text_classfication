# @Time : 2018/7/27 16:48
# @Author : Chicharito_Ron
# @File : classifier.py
# @Software: PyCharm Community Edition

import json
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def res_visual():
    with open('./static/res.json', encoding='utf-8') as f:
        res = json.load(f)

    legends = ['朴素贝叶斯', '逻辑回归', '决策树', '随机森林', 'SVM']

    for i in range(len(res)):
        r = res[i]
        plt.plot(r, marker='*', markersize=5, label=legends[i])

    plt.xlabel('第i次交叉验证')
    plt.ylabel('预测准确率')
    plt.title('文本分类')
    plt.xticks(np.arange(20), np.arange(1, 21))
    plt.legend()
    # plt.savefig('./static/分类结果.jpg', dpi=400, bbox_inches='tight')
    plt.show()


def compare_visual():
    with open('./static/lr.json', encoding='utf-8') as f:
        lr_res = json.load(f)

    fig, axes = plt.subplots(2, 1)

    axes[0].plot(np.arange(1, 21), lr_res[0], marker='*', markersize=5)
    axes[0].plot(np.arange(1, 21), lr_res[1],
                 marker='*', markersize=5, label='降维后结果')
    axes[0].legend()
    axes[0].set_xticks(np.arange(1, 21))
    axes[0].set_title('逻辑回归分类结果')
    axes[0].set_ylabel('预测准确率')

    with open('./static/dt.json', encoding='utf-8') as f:
        dt_res = json.load(f)

    axes[1].plot(np.arange(1, 21), dt_res[0], marker='*', markersize=5)
    axes[1].plot(np.arange(1, 21), dt_res[1],
                 marker='*', markersize=5, label='降维后结果')
    axes[1].legend()
    axes[1].set_xticks(np.arange(1, 21))
    axes[1].set_title('决策树分类结果')
    axes[1].set_ylabel('预测准确率')
    axes[1].set_xlabel('第i次交叉验证')
    plt.subplots_adjust(hspace=0.5)  # 调整subplots的间距
    # plt.savefig('./static/降维前后分类结果.jpg', dpi=400, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # res_visual()
    compare_visual()
