#!/usr/bin/env python3
"""
一般化線形混合モデル（GLMM）チュートリアル

このファイルでは、GLMMの理論と実践について学びます。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'

class GLMMTutorial:
    """
    GLMMの理解を深めるためのチュートリアルクラス
    """
    
    def __init__(self):
        self.data = None
        
    def generate_sample_data(self, n_groups=10, n_obs_per_group=20):
        """
        サンプルデータを生成する
        
        Parameters:
        -----------
        n_groups : int
            グループ数
        n_obs_per_group : int
            各グループの観測数
        """
        np.random.seed(42)
        
        # グループ効果（ランダム効果）
        group_effects = np.random.normal(0, 2, n_groups)
        
        # 固定効果のパラメータ
        beta_0 = 1.5  # 切片
        beta_1 = 0.8  # 説明変数の係数
        
        data_list = []
        
        for i in range(n_groups):
            # 説明変数
            x = np.random.uniform(-2, 2, n_obs_per_group)
            
            # 線形予測子
            linear_predictor = beta_0 + beta_1 * x + group_effects[i]
            
            # ロジスティック回帰の場合（確率）
            prob = 1 / (1 + np.exp(-linear_predictor))
            
            # 二項分布からのサンプリング
            y = np.random.binomial(1, prob)
            
            # データフレームに追加
            for j in range(n_obs_per_group):
                data_list.append({
                    'group': f'Group_{i+1}',
                    'group_id': i,
                    'x': x[j],
                    'y': y[j],
                    'prob': prob[j],
                    'linear_predictor': linear_predictor[j]
                })
        
        self.data = pd.DataFrame(data_list)
        return self.data
    
    def explore_data(self):
        """
        データの探索的分析
        """
        if self.data is None:
            print("データが生成されていません。generate_sample_data()を先に実行してください。")
            return
        
        print("=== データの概要 ===")
        print(f"データサイズ: {self.data.shape}")
        print(f"グループ数: {self.data['group'].nunique()}")
        print(f"総観測数: {len(self.data)}")
        
        print("\n=== 基本統計量 ===")
        print(self.data.describe())
        
        print("\n=== グループ別の統計 ===")
        group_stats = self.data.groupby('group').agg({
            'y': ['count', 'mean', 'std'],
            'x': ['mean', 'std']
        }).round(3)
        print(group_stats)
        
        # 可視化
        self._create_visualizations()
    
    def _create_visualizations(self):
        """
        データの可視化
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. グループ別の応答変数の分布
        sns.boxplot(data=self.data, x='group', y='y', ax=axes[0,0])
        axes[0,0].set_title('グループ別の応答変数の分布')
        axes[0,0].set_xlabel('グループ')
        axes[0,0].set_ylabel('応答変数 (y)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. 説明変数と応答変数の関係
        sns.scatterplot(data=self.data, x='x', y='y', hue='group', alpha=0.6, ax=axes[0,1])
        axes[0,1].set_title('説明変数と応答変数の関係')
        axes[0,1].set_xlabel('説明変数 (x)')
        axes[0,1].set_ylabel('応答変数 (y)')
        
        # 3. グループ別の確率の分布
        sns.boxplot(data=self.data, x='group', y='prob', ax=axes[1,0])
        axes[1,0].set_title('グループ別の確率の分布')
        axes[1,0].set_xlabel('グループ')
        axes[1,0].set_ylabel('確率 (prob)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 線形予測子の分布
        sns.histplot(data=self.data, x='linear_predictor', hue='group', alpha=0.6, ax=axes[1,1])
        axes[1,1].set_title('線形予測子の分布')
        axes[1,1].set_xlabel('線形予測子')
        axes[1,1].set_ylabel('頻度')
        
        plt.tight_layout()
        plt.show()
    
    def explain_glmm_concepts(self):
        """
        GLMMの概念を説明
        """
        print("=== GLMMの基本概念 ===")
        print("\n1. 固定効果（Fixed Effects）:")
        print("   - 全体的な傾向を表すパラメータ")
        print("   - 全てのグループに共通する効果")
        print("   - 例: 年齢、性別、教育レベルなど")
        
        print("\n2. ランダム効果（Random Effects）:")
        print("   - グループ間の変動を表すパラメータ")
        print("   - 各グループ固有の効果")
        print("   - 例: 学校、病院、地域など")
        
        print("\n3. リンク関数（Link Function）:")
        print("   - 応答変数を線形予測子に変換する関数")
        print("   - 例: ロジット関数、プロビット関数、対数関数など")
        
        print("\n4. 分布族（Distribution Family）:")
        print("   - 応答変数の確率分布")
        print("   - 例: 二項分布、ポアソン分布、正規分布など")
        
        print("\n=== GLMMの利点 ===")
        print("1. 相関のあるデータの適切なモデリング")
        print("2. グループ間の変動の考慮")
        print("3. より正確な推論")
        print("4. 過分散の適切な処理")
    
    def demonstrate_glmm_estimation(self):
        """
        GLMMの推定過程をデモンストレーション
        """
        print("=== GLMMの推定過程 ===")
        
        # 簡単な例：ロジスティック回帰
        print("\n1. ロジスティック回帰（固定効果のみ）:")
        print("   g(μ) = log(μ/(1-μ)) = β₀ + β₁x")
        
        # データから推定
        if self.data is not None:
            # 単純なロジスティック回帰
            from sklearn.linear_model import LogisticRegression
            
            X = self.data[['x']].values
            y = self.data['y'].values
            
            lr = LogisticRegression()
            lr.fit(X, y)
            
            print(f"   推定された係数: β₀ = {lr.intercept_[0]:.3f}, β₁ = {lr.coef_[0][0]:.3f}")
            
            # 予測確率
            y_pred_proba = lr.predict_proba(X)[:, 1]
            
            # 適合度の評価
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, y_pred_proba)
            print(f"   AUC: {auc:.3f}")
        
        print("\n2. GLMM（固定効果 + ランダム効果）:")
        print("   g(μᵢⱼ) = log(μᵢⱼ/(1-μᵢⱼ)) = β₀ + β₁xᵢⱼ + bᵢ")
        print("   ここで、bᵢ ~ N(0, σ²ᵦ) はグループiのランダム効果")
        
        print("\n3. 推定方法:")
        print("   - 最尤推定（Maximum Likelihood Estimation）")
        print("   - 制限付き最尤推定（Restricted Maximum Likelihood）")
        print("   - ベイズ推定（Bayesian Estimation）")
    
    def show_interpretation_examples(self):
        """
        GLMMの解釈例を示す
        """
        print("=== GLMMの解釈例 ===")
        
        print("\n1. 固定効果の解釈:")
        print("   β₁ = 0.8 の場合:")
        print("   - 説明変数xが1単位増加すると、")
        print("   - ロジットスケールで0.8増加")
        print("   - オッズ比: exp(0.8) ≈ 2.23")
        print("   - つまり、xが1単位増加すると、成功確率が約2.23倍になる")
        
        print("\n2. ランダム効果の解釈:")
        print("   σ²ᵦ = 4 の場合:")
        print("   - グループ間の変動の大きさ")
        print("   - 95%のグループは、平均から±1.96×√4 = ±3.92の範囲内")
        print("   - グループ間でかなりの変動があることを示す")
        
        print("\n3. 予測の解釈:")
        print("   - 固定効果: 全体的な傾向")
        print("   - ランダム効果: 特定のグループでの調整")
        print("   - 個別予測: 固定効果 + ランダム効果")
    
    def practical_applications(self):
        """
        実用的な応用例
        """
        print("=== GLMMの実用的な応用例 ===")
        
        print("\n1. 医学研究:")
        print("   - 患者の治療効果（患者をグループ化）")
        print("   - 病院間の治療成績の比較")
        print("   - 反復測定データの分析")
        
        print("\n2. 教育研究:")
        print("   - 生徒の学力（学校をグループ化）")
        print("   - クラス間の学習効果の比較")
        print("   - 縦断データの分析")
        
        print("\n3. 社会科学:")
        print("   - 地域間の政策効果の比較")
        print("   - 世帯調査データの分析")
        print("   - パネルデータの分析")
        
        print("\n4. 生態学:")
        print("   - 個体群の生存率（生息地をグループ化）")
        print("   - 種間の相互作用の分析")
        print("   - 時系列データの分析")

def main():
    """
    メイン関数
    """
    print("一般化線形混合モデル（GLMM）チュートリアル")
    print("=" * 50)
    
    # チュートリアルのインスタンスを作成
    tutorial = GLMMTutorial()
    
    # サンプルデータを生成
    print("\n1. サンプルデータの生成...")
    data = tutorial.generate_sample_data()
    print("データ生成完了！")
    
    # データの探索
    print("\n2. データの探索的分析...")
    tutorial.explore_data()
    
    # GLMMの概念説明
    print("\n3. GLMMの概念説明...")
    tutorial.explain_glmm_concepts()
    
    # GLMMの推定過程
    print("\n4. GLMMの推定過程...")
    tutorial.demonstrate_glmm_estimation()
    
    # 解釈例
    print("\n5. GLMMの解釈例...")
    tutorial.show_interpretation_examples()
    
    # 実用的な応用例
    print("\n6. 実用的な応用例...")
    tutorial.practical_applications()
    
    print("\n" + "=" * 50)
    print("チュートリアル完了！")
    print("GLMMについての理解が深まりましたか？")

if __name__ == "__main__":
    main()