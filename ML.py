"""
这主要用于使用不同的机器学习方法，解决分类问题，并评估它们之间性能的不同
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class ModelTrainer:
    
    def __init__(self):
        
        pass

    def split_data(self, X, y):
        """
        简单的划分数据集操作
        :param X: 样本数据，n行k列
        :param y: 类别变量，n维
        :return: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 42)
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        输入训练数据和测试数据，返回多种分类器预测的评估结果
        :return: <dict> {'FPR': ?, 'FNR': ?, 'Precision': ?, 'Recall': ?, 'Accuracy': ?, 'F_\beta*': ?}
        """
        classifiers = {
            "SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier()
        }
    
        results = {}
    
        for name, clf in classifiers.items():
            # 训练模型
            clf.fit(X_train, y_train)
    
            # 预测
            y_pred = clf.predict(X_test)
    
            # 评估
            tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

            # 计算评价指标
            fpr = fp / (fp + tn)  # 假阳性率
            fnr = fn / (fn + tp)  # 假阴性率
            precision = tp / (tp + fp)  # 精确率
            recall = tp / (fn + tp)  # 召回率
            accuracy = (tp+tn)/(tp+tn+fp+fn) # 准确率

            
            results[name] = {'FPR': fpr, 'FNR': fnr, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}
            
            # F-score,beta 衡量 recall与 precision的权重，beta越大，recall越重要
            beta = [1,2,0.5]
            for b in beta:
                key_name = f'F{b}'
                key_value = (1+b**2)*(recall * precision)/(b**2*precision + recall)
                
                results[name][key_name] = key_value
    
        return results
        
 
# # 调用函数例子
# results = train_and_evaluate(X_train, X_test, y_train, y_test)

# # 可视化
# accuracies = [results[name]["accuracy"] for name in results]
# names = list(results.keys())

# plt.bar(names, accuracies)
# plt.xlabel('Classifier')
# plt.ylabel('Accuracy')
# plt.title('Comparison of Different Classifiers')
# plt.show()
 