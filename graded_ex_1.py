import pandas as pd

class DataAnalysis:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def select_variable(self, variable_name):
        """选择指定变量"""
        if variable_name in self.data.columns:
            return self.data[variable_name]
        else:
            raise ValueError("变量不存在")

    def check_normality(self, variable_name):
        """检查指定变量的正态性"""
        from scipy import stats
        _, p_value = stats.shapiro(self.data[variable_name])
        return p_value >= 0.05  # 如果p值>=0.05，则为正态分布

    def hypothesis_test_anova(self, group_col, target_col):
        """进行ANOVA检验"""
        from scipy import stats
        groups = [group[1][target_col].values for group in self.data.groupby(group_col)]
        return stats.f_oneway(*groups)

    def hypothesis_test_kruskal(self, group_col, target_col):
        """进行Kruskal-Wallis检验"""
        from scipy import stats
        groups = [group[1][target_col].values for group in self.data.groupby(group_col)]
        return stats.kruskal(*groups)

