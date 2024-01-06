import pandas as pd
from plotnine import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CPLOT():
    """
    绘制分类图表
    """
    def __init__(self,dataframe, index_name = 'index', value_name = 'value'):
        """
        :param dataframe: 数据框
        :param index_name: x-axis名称
        :param value_name: y-axis名称
        """
        self.df = dataframe
        self.index_name = index_name
        self.value_name = value_name

    def multi_bar_plot(self):
        """
        绘制复式条形图
        :return:作出图像
        """
        dataframe = self.df
        index_name = self.index_name
        value_name = self.value_name
        # 转换为长格式
        df_long = dataframe.reset_index().melt(id_vars='index', var_name=index_name, value_name=value_name)

        # 创建复式条形图
        plot = (ggplot(df_long, aes(x=index_name, y=value_name, fill='index')) +
                geom_bar(stat='identity',color = 'black', position='dodge') +
                theme(axis_text_x=element_text(angle=45, hjust=1),figure_size = (10,5)) +
                labs(x=index_name, y=value_name, fill='Metric'))

        # 显示图表
        print(plot)

    def lollipop_plot(self):
        """
        绘制棒棒糖图
        :return:作出图像
        """
        dataframe = self.df
        index_name = self.index_name
        value_name = self.value_name
        # 转换为长格式
        df_long = dataframe.reset_index().melt(id_vars='index', var_name=index_name, value_name=value_name)

        # 创建棒棒糖图
        plot = (ggplot(df_long, aes(x=value_name, y=index_name, fill='index')) +
                geom_line(aes(group=index_name)) +
                geom_point(shape='o', size=5, color='k')+
                theme(axis_text_x=element_text(angle=45, hjust=1),figure_size = (10,5)) +
                labs(x=value_name, y=index_name, fill='Metric'))

        print(plot)

    def heatmap_plot(self):
        """
        绘制热力图
        :return:作出图像
        """
        df = self.df
        plt.figure(dpi = 100)
        sns.heatmap(df,cmap='YlGnBu',annot=True)
        plt.title('heatmap')
        plt.show()

    def Nightingale_rose_plot(self):
        """
        绘制南丁格尔玫瑰图
        :return: 作出图像
        """
        dataframe = self.df
        index_name = self.index_name
        value_name = self.value_name
        # 转换为长格式
        df_long = dataframe.reset_index().melt(id_vars='index', var_name=index_name, value_name=value_name)

        # 创建南丁格尔玫瑰图
        num_vars = len(df_long['index'].unique())
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # 绘制每个分类器的图
        for classifier in df_long[index_name].unique():
            values = df_long[df_long[index_name] == classifier][value_name].tolist()
            values += values[:1]
            ax.fill(angles, values, alpha=0.25, label=classifier)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(df_long['index'].unique())

        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.show()

    def radar_plot(self):
        """
        绘制雷达图
        :return : 作出图像
        """
        dataframe = self.df
        labels = dataframe.columns.values
        num_vars = len(labels)
        
        # 计算每个角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        # 绘制每个指标的雷达图
        for idx, row in dataframe.iterrows():
            values = row.values.tolist()
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, label=idx, linewidth=1.5)  # 绘制线条
            ax.fill(angles, values, alpha=0.25)  # 填充颜色
        
        # 设置雷达图的标签
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        
        # 添加图例
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        # 显示图形
        plt.show()
        

