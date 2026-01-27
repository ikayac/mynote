#!/usr/bin/env python3
"""
釣果尾数のGLMM例 - 気温（固定効果）と釣りの腕前（ランダム効果）

このファイルでは、釣果尾数が気温に依存し、釣りの腕前によって個人差がある
ポアソン分布に従うGLMMの例を詳しく説明します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FishingGLMMExample:
    """
    釣果尾数のGLMM例を詳しく説明するクラス
    """
    
    def __init__(self):
        self.data = None
        self.true_params = {}
        
    def create_fishing_data(self, n_anglers=20, n_trips=10):
        """
        釣果尾数のサンプルデータを作成
        
        Parameters:
        -----------
        n_anglers : int
            釣り人の数
        n_trips : int
            各釣り人の釣行回数
        """
        np.random.seed(42)
        
        # 真のパラメータ（後で手計算で確認するため）
        self.true_params = {
            'beta_0': 1.5,      # 切片（固定効果）
            'beta_1': 0.3,      # 気温の係数（固定効果）
            'sigma_b': 0.8      # ランダム効果の標準偏差
        }
        
        # 釣り人固有の効果（ランダム効果）
        # b_i ~ N(0, σ²_b) ここで σ²_b = 0.8² = 0.64
        angler_effects = np.random.normal(0, self.true_params['sigma_b'], n_anglers)
        
        # 気温の範囲（-5度から25度）
        temp_range = np.linspace(-5, 25, n_trips)
        
        data_list = []
        
        for i in range(n_anglers):
            for j, temp in enumerate(temp_range):
                # 線形予測子（対数スケール）
                # η_ij = β₀ + β₁ × temp_ij + b_i
                linear_predictor = (self.true_params['beta_0'] + 
                                  self.true_params['beta_1'] * temp + 
                                  angler_effects[i])
                
                # 期待値（指数関数で変換）
                # λ_ij = exp(η_ij) = exp(β₀ + β₁ × temp_ij + b_i)
                lambda_param = np.exp(linear_predictor)
                
                # ポアソン分布からのサンプリング
                # y_ij ~ Poisson(λ_ij)
                catch_count = np.random.poisson(lambda_param)
                
                data_list.append({
                    'angler_id': i,
                    'angler': f'Angler_{i+1}',
                    'trip': j + 1,
                    'temperature': temp,
                    'catch_count': catch_count,
                    'lambda_param': lambda_param,
                    'linear_predictor': linear_predictor,
                    'angler_effect': angler_effects[i]
                })
        
        self.data = pd.DataFrame(data_list)
        return self.data
    
    def explain_glmm_structure(self):
        """
        GLMMの構造を詳しく説明
        """
        print("=== 釣果尾数のGLMMの構造 ===")
        print("\n1. モデルの数式:")
        print("   y_ij ~ Poisson(λ_ij)")
        print("   log(λ_ij) = β₀ + β₁ × temp_ij + b_i")
        print("   b_i ~ N(0, σ²_b)")
        
        print("\n2. パラメータの意味:")
        print("   - y_ij: 釣り人iのj回目の釣行での釣果尾数")
        print("   - temp_ij: 釣り人iのj回目の釣行での気温")
        print("   - β₀: 切片（固定効果）")
        print("   - β₁: 気温の係数（固定効果）")
        print("   - b_i: 釣り人iのランダム効果（釣りの腕前など）")
        print("   - σ²_b: ランダム効果の分散")
        
        print("\n3. 真のパラメータ値:")
        print(f"   - β₀ = {self.true_params['beta_0']}")
        print(f"   - β₁ = {self.true_params['beta_1']}")
        print(f"   - σ_b = {self.true_params['sigma_b']}")
        print(f"   - σ²_b = {self.true_params['sigma_b']**2:.3f}")
    
    def demonstrate_hand_calculation(self):
        """
        手計算で考え方を追う
        """
        print("\n=== 手計算による考え方の過程 ===")
        
        if self.data is None:
            print("データが生成されていません。create_fishing_data()を先に実行してください。")
            return
        
        # 特定の釣り人と釣行を選んで計算例を示す
        example_data = self.data[(self.data['angler_id'] == 0) & (self.data['trip'] == 1)].iloc[0]
        
        print(f"\n1. 具体例（Angler_1の1回目の釣行）:")
        print(f"   気温: {example_data['temperature']:.1f}度")
        print(f"   釣り人のランダム効果: {example_data['angler_effect']:.3f}")
        
        print(f"\n2. 線形予測子の計算:")
        print(f"   η = β₀ + β₁ × temp + b_i")
        print(f"   η = {self.true_params['beta_0']} + {self.true_params['beta_1']} × {example_data['temperature']:.1f} + {example_data['angler_effect']:.3f}")
        
        linear_pred = self.true_params['beta_0'] + self.true_params['beta_1'] * example_data['temperature'] + example_data['angler_effect']
        print(f"   η = {linear_pred:.3f}")
        
        print(f"\n3. 期待値の計算:")
        print(f"   λ = exp(η) = exp({linear_pred:.3f})")
        lambda_val = np.exp(linear_pred)
        print(f"   λ = {lambda_val:.3f}")
        
        print(f"\n4. 実際の釣果尾数:")
        print(f"   y ~ Poisson({lambda_val:.3f})")
        print(f"   実際の値: {example_data['catch_count']}")
        
        print(f"\n5. 他の釣り人との比較:")
        print("   同じ気温でも、釣り人の腕前（ランダム効果）によって期待値が変わる")
        
        # 同じ気温での異なる釣り人の比較
        same_temp_data = self.data[self.data['temperature'] == example_data['temperature']].head(5)
        
        print(f"\n   気温{example_data['temperature']:.1f}度での各釣り人の期待値:")
        for _, row in same_temp_data.iterrows():
            angler_lambda = np.exp(row['linear_predictor'])
            print(f"   Angler_{row['angler_id']+1}: λ = {angler_lambda:.3f} (b_i = {row['angler_effect']:.3f})")
    
    def show_parameter_interpretation(self):
        """
        パラメータの解釈を説明
        """
        print("\n=== パラメータの解釈 ===")
        
        print("\n1. 固定効果の解釈:")
        print(f"   - β₀ = {self.true_params['beta_0']}: 気温0度での平均釣果尾数の対数")
        print(f"   - β₁ = {self.true_params['beta_1']}: 気温が1度上がると、釣果尾数が{np.exp(self.true_params['beta_1']):.2f}倍になる")
        
        print(f"\n2. ランダム効果の解釈:")
        print(f"   - σ_b = {self.true_params['sigma_b']}: 釣り人の腕前による変動の大きさ")
        print(f"   - 95%の釣り人は、平均から±{1.96 * self.true_params['sigma_b']:.2f}の範囲内")
        
        print(f"\n3. 具体例での解釈:")
        temp_effect = np.exp(self.true_params['beta_1'])
        print(f"   - 気温が5度から10度に上がると、釣果尾数は{temp_effect**5:.2f}倍になる")
        print(f"   - 気温が10度から15度に上がると、釣果尾数は{temp_effect**5:.2f}倍になる")
    
    def demonstrate_likelihood_calculation(self):
        """
        尤度計算の過程を説明
        """
        print("\n=== 尤度計算の過程 ===")
        
        if self.data is None:
            print("データが生成されていません。")
            return
        
        # 特定の釣り人のデータを選ぶ
        angler_data = self.data[self.data['angler_id'] == 0]
        
        print(f"\n1. Angler_1のデータ:")
        print(f"   釣行回数: {len(angler_data)}")
        print(f"   釣果尾数: {angler_data['catch_count'].tolist()}")
        
        print(f"\n2. 条件付き尤度（ランダム効果b₁が与えられた場合）:")
        print("   f(y₁|b₁) = ∏ᵢ Poisson(y₁ᵢ|λ₁ᵢ)")
        print("   ここで λ₁ᵢ = exp(β₀ + β₁ × temp₁ᵢ + b₁)")
        
        # 条件付き尤度の計算
        conditional_likelihood = 1.0
        for _, row in angler_data.iterrows():
            lambda_val = np.exp(row['linear_predictor'])
            y_val = row['catch_count']
            poisson_prob = stats.poisson.pmf(y_val, lambda_val)
            conditional_likelihood *= poisson_prob
            print(f"   - 釣行{row['trip']}: y = {y_val}, λ = {lambda_val:.3f}, P(y|λ) = {poisson_prob:.6f}")
        
        print(f"\n   条件付き尤度: {conditional_likelihood:.2e}")
        
        print(f"\n3. ランダム効果の事前分布:")
        print(f"   f(b₁) = N(b₁|0, {self.true_params['sigma_b']**2:.3f})")
        
        angler_effect = angler_data.iloc[0]['angler_effect']
        prior_prob = stats.norm.pdf(angler_effect, 0, self.true_params['sigma_b'])
        print(f"   b₁ = {angler_effect:.3f}での確率密度: {prior_prob:.3f}")
        
        print(f"\n4. 完全な尤度:")
        print("   f(y₁, b₁) = f(y₁|b₁) × f(b₁)")
        complete_likelihood = conditional_likelihood * prior_prob
        print(f"   完全な尤度: {complete_likelihood:.2e}")
        
        print(f"\n5. 全データの尤度:")
        print("   f(y, b) = ∏ᵢ f(yᵢ|bᵢ) × f(bᵢ)")
        print("   この積分を最大化することでパラメータを推定")
    
    def show_data_visualization(self):
        """
        データの可視化
        """
        if self.data is None:
            print("データが生成されていません。")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 気温と釣果尾数の関係（全体）
        sns.scatterplot(data=self.data, x='temperature', y='catch_count', 
                       hue='angler', alpha=0.7, ax=axes[0,0])
        axes[0,0].set_title('気温と釣果尾数の関係（全体）')
        axes[0,0].set_xlabel('気温（度）')
        axes[0,0].set_ylabel('釣果尾数')
        
        # 2. 釣り人別の釣果尾数の分布
        sns.boxplot(data=self.data, x='angler', y='catch_count', ax=axes[0,1])
        axes[0,1].set_title('釣り人別の釣果尾数の分布')
        axes[0,1].set_xlabel('釣り人')
        axes[0,1].set_ylabel('釣果尾数')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. 気温別の平均釣果尾数
        temp_means = self.data.groupby('temperature')['catch_count'].agg(['mean', 'std']).reset_index()
        axes[1,0].plot(temp_means['temperature'], temp_means['mean'], 'ro-', linewidth=2)
        axes[1,0].fill_between(temp_means['temperature'], 
                              temp_means['mean'] - temp_means['std'],
                              temp_means['mean'] + temp_means['std'], 
                              alpha=0.3)
        axes[1,0].set_title('気温別の平均釣果尾数')
        axes[1,0].set_xlabel('気温（度）')
        axes[1,0].set_ylabel('平均釣果尾数')
        
        # 4. ランダム効果の分布
        angler_effects = self.data.groupby('angler_id')['angler_effect'].first()
        axes[1,1].hist(angler_effects, bins=10, alpha=0.7, edgecolor='black')
        axes[1,1].axvline(0, color='red', linestyle='--', label='平均=0')
        axes[1,1].set_title('釣り人のランダム効果の分布')
        axes[1,1].set_xlabel('ランダム効果')
        axes[1,1].set_ylabel('頻度')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_model_comparison(self):
        """
        モデルの比較を説明
        """
        print("\n=== モデルの比較 ===")
        
        print("\n1. 単純なポアソン回帰（ランダム効果なし）:")
        print("   log(λ_ij) = β₀ + β₁ × temp_ij")
        print("   問題: 釣り人内の相関を無視")
        
        print("\n2. GLMM（ランダム効果あり）:")
        print("   log(λ_ij) = β₀ + β₁ × temp_ij + b_i")
        print("   利点: 釣り人内の相関を適切にモデリング")
        
        print("\n3. 分散成分の分解:")
        print("   - 固定効果による変動: 気温の影響")
        print("   - ランダム効果による変動: 釣り人の腕前の違い")
        print("   - 残差変動: その他の要因")
        
        print("\n4. モデル選択の基準:")
        print("   - AIC: モデルの複雑さと適合度のバランス")
        print("   - BIC: サンプルサイズを考慮した選択")
        print("   - 残差分析: モデルの妥当性の確認")

def main():
    """
    メイン関数
    """
    print("釣果尾数のGLMM例 - 気温（固定効果）と釣りの腕前（ランダム効果）")
    print("=" * 70)
    
    # 例のインスタンスを作成
    example = FishingGLMMExample()
    
    # サンプルデータの作成
    print("\n1. サンプルデータの作成...")
    data = example.create_fishing_data()
    print("データ作成完了！")
    
    # GLMMの構造説明
    print("\n2. GLMMの構造説明...")
    example.explain_glmm_structure()
    
    # 手計算による説明
    print("\n3. 手計算による説明...")
    example.demonstrate_hand_calculation()
    
    # パラメータの解釈
    print("\n4. パラメータの解釈...")
    example.show_parameter_interpretation()
    
    # 尤度計算の過程
    print("\n5. 尤度計算の過程...")
    example.demonstrate_likelihood_calculation()
    
    # データの可視化
    print("\n6. データの可視化...")
    example.show_data_visualization()
    
    # モデルの比較
    print("\n7. モデルの比較...")
    example.demonstrate_model_comparison()
    
    print("\n" + "=" * 70)
    print("説明完了！")
    print("釣果尾数のGLMMについて理解が深まりましたか？")

if __name__ == "__main__":
    main()