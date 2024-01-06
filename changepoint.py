# 该类用于电弧检测的变电检测方法
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from PIL import Image

class CPD():
    def __init__(self):
        self.ctime = self.current_time()
        self.savedir = os.path.join('savefigure',self.ctime)
    # 该函数可以实现变点检测及画图、保存图

    def current_time(self):
        """
        获取当前时间，格式为：year-month-day hour-minute-second
        :return: str
        """
        timestamp = int(time.time())
    
        date: str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    
        date: str = date.replace(':', '-')
    
        return date

    def hampel_filter(self, data_stream, initial_window_size, n_sigmas=3, 
                                                     outlier_sequence_threshold=3,
                                                     inlier_sequence_threshold = 70,
                                                     is_save = True,
                                                     ):
        """
        该函数使用hampel滤波器的思想，实现实时数据变点检测以及数据的可视化
        :param: data_stream (iterable): 数据流的迭代器
        :param: initial_window_size (int): 初始窗口大小
        :param: n_sigmas (int, optional): 标准误差，用以检测离群点，默认为3
        :param: outlier_sequence_threshold (int, optional): 对连续离群点的容忍度，超出视为变点，默认为3
        :param: inlier_sequence_threshold (int, optional): 在异常数据流情形，对连续非异常点的容忍度，超出视为变点，默认为70
        :param: is_save(bool): 是否储存所有图片
        :return: tuple,(new_data_point, is_outlier, is_changepoint),数据流中的每个点
        
        """
        k = 1.4826  # 高斯分布的比例因子
        consecutive_inliers = 0
        consecutive_outliers = 0
        last_changepoint = 0
        data_history = []
        convinced_data = []    
        change_point = []
        state = 0    # 0表示正常，1表示异常
        temp_data = []
        outliers = []
    
        # 创建文件夹，保存图表数据
        ctime = self.current_time()
        try:
            if is_save:
                os.makedirs(self.savedir, exist_ok=True)
        except OSError as e:
            print(f"创建目录时出错: {e}")
            return
    
        # for循环传入数据
        for i, new_point in enumerate(data_stream):
            data_history.append(new_point)
            convinced_data.append(new_point)
    
            # 数据到初始窗口大小一时，开始筛除内部的离群点
            if int(initial_window_size / 2) <= len(convinced_data) < initial_window_size:
                median = np.median(convinced_data)
                mad = k * np.median(np.abs(convinced_data - median))
                threshold = n_sigmas * mad
                outlier_indices = [idx for idx, point in enumerate(convinced_data) if np.abs(point - median) > threshold]
    
                if outlier_indices:
                    # 仅移除最大的一个离群点
                    max_outlier_idx = max(outlier_indices, key=lambda idx: np.abs(convinced_data[idx] - median))
                    max_outlier_value = convinced_data[max_outlier_idx]
                    del convinced_data[max_outlier_idx]
        
                    # 保存离群点信息（在data_stream中的索引和值）
                    outliers.append((i - len(convinced_data) + max_outlier_idx, max_outlier_value))
                    is_outlier = True
                    outlier_info = (i - len(convinced_data) + max_outlier_idx, max_outlier_value)
                else:
                    is_outlier = False
                    outlier_info = None
    
            # 数据达到初始窗口大小之前，不检测新点是否是变点
            if len(convinced_data) <= initial_window_size:
                yield (i,new_point, False, False)
                continue      
            
            # MAD
            try:
                median = np.median(convinced_data)
                mad = k * np.median(np.abs(convinced_data - median))
            except Exception as e:
                print(f"数学运算时出错: {e}")
    
            if np.abs(new_point - median) > n_sigmas * mad:
                is_outlier = True
    
            # 如果当前时序数据正常，则需连续出现"consecutive_outliers"个异常值点，则认为第一个时出现变点
            if state == 0:
                
                if np.abs(new_point - median) > n_sigmas * mad:
                    is_outlier = True
                    consecutive_outliers += 1
                    temp_data.append(convinced_data.pop())
                else:
                    is_outlier = False
                    consecutive_outliers = 0
                    convinced_data += temp_data
                    temp_data = []
        
                if consecutive_outliers >= outlier_sequence_threshold:
                    is_changepoint = True
                    state = 1 - state
                    last_changepoint = i - outlier_sequence_threshold + 1
                    change_point.append(last_changepoint)
                    temp_data = []
                    consecutive_outliers = 0
                else:
                    is_changepoint = False
                    
            # 如果当前时序数据异常，则需连续出现"consecutive_inliers"个非异常值点，则认为第一个时出现变点
            elif state == 1:
                
                if np.abs(new_point - median) <= n_sigmas * mad:
                    is_outlier = False
                    is_inlier = True
                    consecutive_inliers += 1
                    temp_data.append(convinced_data.pop())
                    
                else:
                    is_outlier = True
                    is_inlier = False
                    consecutive_inliers = 0
                    temp_data = []
    
                if consecutive_inliers >= inlier_sequence_threshold:
                    is_changepoint = True
                    state = 1 - state
                    last_changepoint = i - inlier_sequence_threshold + 1
                    change_point.append(last_changepoint)
    
                    convinced_data += temp_data
                    temp_data = []         
                    consecutive_inliers = 0
                else:
                    is_changepoint = False
    
            # 变点检测效果可视化
            plt.figure(figsize=(10, 4))
            if not outliers:
                for loc,outlier_point in enumerate(outliers):
                    if loc == 1:
                        plt.scatter(outlier_point[0],outlier_point[1],color = 'grey',label = '忽略的离群点')
                    else:
                        plt.scatter(outlier_point[0],outlier_point[1],color = 'grey')
            plt.scatter(i, new_point, color='red' if is_outlier else 'green', label='Current Point')
    
            # 设置背景颜色块
            colors = ['green', 'red']
            current_color_index = 0
            start = 0
            for end in change_point:
                plt.axvspan(start, end, color=colors[current_color_index], alpha=0.3)
                plt.plot(range(start, end), data_history[start:end], color=colors[current_color_index])  # 绘制折线的对应部分
                start = end
                current_color_index = 1 - current_color_index  # 切换颜色
        
            # 处理最后一个区间
            if len(data_history) > start:
                plt.axvspan(start, len(data_history), color=colors[current_color_index], alpha=0.3)
                plt.plot(range(start, len(data_history)), data_history[start:len(data_history)],   color=colors[current_color_index])  # 绘制折线的对应部分
    
            
            for num,cp in enumerate(change_point):
                if num == 0:
                    plt.axvline(x=cp, color='red', linestyle='--', label='Changepoint')
                else:
                    plt.axvline(x=cp, color='red', linestyle='--')
                
            
            plt.title(f"Hampel Filter变点检测({i+1} 个点)")
            plt.xlabel('窗口期')
            plt.ylabel('feature')
            plt.legend()
            try:
                if is_save:
                    plt.savefig(f"{self.savedir}/realtime_plot_{i+1}.png")
            except Exception as e:
                print(f"保存图像时出错: {e}")
            if i == len(data_stream) - 1:
                plt.show()
            else:
                plt.close()
    
            yield (i,new_point, is_outlier, is_changepoint)

    def merge_to_gif(self,duration = 100,loop = 0):
        """
        把一个文件夹内的所有图片格式文件合成gif
        :param duration: 每张图片播放时间间隔，单位毫秒，1000ms = 1s
        :param loop: k表示循环k次，0表示一直循环
        :return: None
        """
        dirpath = self.savedir
        def extract_number(filename):
            """
            提取文件名中的数字部分并转换为整数
            """
            # 假设文件名格式为 "realtime_plot_数字.png"
            number_part = filename.split('_')[-1].split('.')[0]  # 分割字符串并获取数字部分
            return int(number_part)
    
        files = os.listdir(dirpath)
        pic_files = [i for i in files if (i.split('.')[-1]) in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']]
        pic_files.sort(key=extract_number)
    
        # 将所有图片合成gif图
        output_name = "merge_to_gif.gif"  # gif图像命名
    
        try:
            images = [Image.open(os.path.join(dirpath, file)) for file in pic_files]
            images[0].save(os.path.join(dirpath, output_name), save_all=True, append_images=images[1:], duration=duration, loop=loop)
        except Exception as e:
            print(f"生成 GIF 时出错: {e}")

    def generate_state_sequence(self, change_points, length, initial_state = 0):
        """
        根据变点索引、初始状态和长度生成状态序列。
    
        :param change_points: 变点的列表，每个变点是一个元组，包含索引和值
        :param initial_state: 初始状态（0或1）
        :param length: 生成的序列长度
        :return: 状态序列列表
        """
        sequence = [initial_state] * length
        for index, _ in change_points:
            if index < length:
                # 在变点处切换状态
                sequence[index:] = [1 - sequence[index]] * (length - index)
    
        return sequence

    def calculate_performance_metrics(self, true_values, predicted_values):
        """
        计算算法评价指标：FPR（假阳性率）、FNR（假阴性率）、Precision（精确率）和Recall（召回率）。
    
        :param true_values: 真实值列表
        :param predicted_values: 预测值列表
        :return: 字典形式的评价指标
        """
        tn, fp, fn, tp = confusion_matrix(true_values, predicted_values).ravel()
    
        # 计算评价指标
        fpr = fp / (fp + tn)  # 假阳性率
        fnr = fn / (fn + tp)  # 假阴性率
        precision = tp / (tp + fp)  # 精确率
        recall = tp / (fn + tp)  # 召回率
        accuracy = (tp+tn)/(tp+tn+fp+fn)  # 准确率

        results = {'FPR': fpr, 'FNR': fnr, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}

        # F-score,beta 衡量 recall与 precision的权重，beta越大，recall越重要
        beta = [1,2,0.5]
        for b in beta:
            key_name = f'F{b}'
            key_value = (1+b**2)*(recall * precision)/(b**2*precision + recall)
            
            results[key_name] = key_value
    
        return results

