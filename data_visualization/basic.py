import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import iplot
from pandas.plotting import radviz
from bubbly.bubbly import bubbleplot
from sklearn.datasets import load_iris
from pandas.plotting import andrews_curves

from wordcloud import WordCloud, STOPWORDS


def load_data():
    iris = load_iris()
    data =  pd.DataFrame(np.concatenate((iris['data'], np.expand_dims(iris['target'],axis=1)),axis =1))
    data.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
    data['Id'] = np.array([i for i in range(len(data))])
    return data


def BubblePlot():
    # 气泡图
    data = load_data()
    figure = bubbleplot(dataset=data,
                        x_column='SepalLengthCm',
                        y_column='PetalLengthCm',
                        z_column='SepalWidthCm',
                        bubble_column='Id',
                        size_column='PetalWidthCm',
                        color_column='Species',
                        x_title='SepalLength(Cm)',
                        y_title='PetalLength(Cm)',
                        z_title='SepalWidth(Cm)',
                        title='IRIS Visualization',
                        x_logscale=False,
                        scale_bubble=0.1,
                        height=600)
    iplot(figure, config={'scrollzoom': True})
    plt.show()
    return

def ScatterPlot():
    # 散点图
    data = load_data()
    fg = sns.FacetGrid(data,hue='Species',size=10)
    fg.map(plt.scatter,"SepalLengthCm", "SepalWidthCm")
    fg.add_legend()
    plt.show()
    return

def HistPlot():
    # 直方图
    data = load_data()
    fg = sns.FacetGrid(data, col='Species')
    fg.map(plt.hist,'SepalWidthCm')
    plt.show()
    return


def KDEPlot():
    # KDE(核密度估计)
    # 对直方图对一种平滑
    data = load_data()
    fg = sns.FacetGrid(data, hue='Species', size=10)
    fg.map(sns.kdeplot,"PetalLengthCm")
    fg.add_legend()
    plt.show()
    return

def BoxPlot():
    # 箱线图-5个黑线是最大值、最小值、中位数和两个四分位数
    data = load_data()
    ax = sns.boxplot(data=data, x="Species", y="PetalLengthCm")
    ax = sns.stripplot(x="Species", y="PetalLengthCm", data=data, jitter=True)
    plt.show()
    return

def ViolinPlot():
    # 小提琴图-作用与箱线图类似
    data = load_data()
    sns.violinplot(data=data, x="Species", y="PetalLengthCm", size=6)
    plt.show()
    return

def AndrewsPlot():
    # 安德鲁斯曲线-样本被转换成一条线,样本越相似曲线越靠近
    data = load_data()
    andrews_curves(data.drop("Id", axis=1), "Species")
    return

def RadvizPlot():
    # 降维可视化
    data = load_data()
    radviz(data.drop("Id", axis=1), "Species")
    return

def HeatMap():
    # 特征间相关系数热力图
    data = load_data()
    sns.heatmap(data.drop("Id", axis=1).corr(),annot=True)
    return

def WordCloud(text_list):
    # 词云
    """
    text_list = [
        "It never once occurred to me that the fumbling might be a mere mistake."
        "Herbert West needed fresh bodies because his life work was the reanimation of the dead."
    ]
    """
    plt.figure(figsize=(16,13))
    wc = WordCloud(background_color="black",
                   max_words=10000,
                   stopwords=STOPWORDS,
                   max_font_size= 40)
    wc.generate(" ".join(text_list))
    plt.title("word cloud", fontsize=20)
    plt.imshow(wc.recolor(colormap= 'Pastel2' , random_state=17), alpha=0.98)
    plt.axis('off')
    return
