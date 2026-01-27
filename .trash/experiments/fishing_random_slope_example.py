#!/usr/bin/env python3
"""
釣果尾数のランダム傾きGLMM例

このファイルでは、切片だけでなく係数にもランダム効果を入れる
ランダム傾きモデルを実装します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FishingRandomSlopeGLMM:
    """
    釣果尾数のランダム傾きGLMM例を詳しく説明するクラス
    """
    
    def __init__(self):
        self.data = None
        self.true_params = {}
        self.random_effects = None
        
    def create_random_slope_data(self, n_anglers=25, n_trips=12):
        """
        ランダム傾きモデルのサンプルデータを作成
        
        Parameters:
        -----------
        n_anglers : int
            釣り人の数
        n_trips : int
            各釣り人の釣行回数
        """
        np.random.seed(42)
        
        # 真のパラメータ
        self.true_params = {
            'beta_0': 1.2,      # 固定切片
            'beta_1': 0.25,     # 固定気温係数
            'sigma_b0': 0.6,    # ランダム切片の標準偏差
            'sigma_b1': 0.15,   # ランダム傾きの標準偏差
            'rho': 0.3          # ランダム切片とランダム傾きの相関
        }
        
        # 共分散行列の構築
        sigma_b0 = self.true_params['sigma_b0']
        sigma_b1 = self.true_params['sigma_b1']
        rho = self.true_params['rho']
        
        # 共分散行列 G
        G = np.array([
            [sigma_b0**2, rho * sigma_b0 * sigma_b1],
            [rho * sigma_b0 * sigma_b1, sigma_b1**2]
        ])
        
        print(f"真の共分散行列 G:")
        print(f"G = {G}")
        
        # ランダム効果の生成（多変量正規分布）
        # (b₀ᵢ, b₁ᵢ) ~ N(0, G)
        self.random_effects = np.random.multivariate_normal(
            mean=[0, 0], 
            cov=G, 
            size=n_anglers
        )
        
        # 気温の範囲（-5度から30度）
        temp_range = np.linspace(-5, 30, n_trips)
        
        data_list = []
        
        for i in range(n_anglers):
            b0_i, b1_i = self.random_effects[i]
            
            for j, temp in enumerate(temp_range):
                # 線形予測子（対数スケール）
                # η_ij = β₀ + β₁ × temp_ij + b₀ᵢ + b₁ᵢ × temp_ij
                linear_predictor = (self.true_params['beta_0'] + 
                                  self.true_params['beta_1'] * temp + 
                                  b0_i + b1_i * temp)
                
                # 期待値（指数関数で変換）
                # λ_ij = exp(η_ij)
                lambda_param = np.exp(linear_predictor)
                
                # ポアソン分布からのサンプリング
                catch_count = np.random.poisson(lambda_param)
                
                data_list.append({
                    'angler_id': i,
                    'angler': f'Angler_{i+1}',
                    'trip': j + 1,
                    'temperature': temp,
                    'catch_count': catch_count,
                    'lambda_param': lambda_param,
                    'linear_predictor': linear_predictor,
                    'random_intercept': b0_i,
                    'random_slope': b1_i,
                    'total_intercept': self.true_params['beta_0'] + b0_i,
                    'total_slope': self.true_params['beta_1'] + b1_i
                })
        
        self.data = pd.DataFrame(data_list)
        return self.data
    
    def explain_random_slope_model(self):
        """
        ランダム傾きモデルの構造を詳しく説明
        """
        print("=== ランダム傾きモデルの構造 ===")
        
        print("\n1. モデルの数式:")
        print("   y_ij ~ Poisson(λ_ij)")
        print("   log(λ_ij) = β₀ + β₁ × temp_ij + b₀ᵢ + b₁ᵢ × temp_ij")
        print("   (b₀ᵢ, b₁ᵢ) ~ N(0, G)")
        
        print("\n2. パラメータの意味:")
        print("   - β₀: 固定切片（全体的な基本釣果尾数の対数）")
        print("   - β₁: 固定気温係数（全体的な気温の効果）")
        print("   - b₀ᵢ: 釣り人iのランダム切片（基本能力の個人差）")
        print("   - b₁ᵢ: 釣り人iのランダム傾き（気温効果の個人差）")
        
        print("\n3. 共分散行列Gの意味:")
        print("   - σ²₀₀: ランダム切片の分散（基本能力のばらつき）")
        print("   - σ²₁₁: ランダム傾きの分散（気温効果のばらつき）")
        print("   - σ₀₁: ランダム切片とランダム傾きの共分散（相関）")
        
        print(f"\n4. 真のパラメータ値:")
        print(f"   - β₀ = {self.true_params['beta_0']}")
        print(f"   - β₁ = {self.true_params['beta_1']}")
        print(f"   - σ_b₀ = {self.true_params['sigma_b0']}")
        print(f"   - σ_b₁ = {self.true_params['sigma_b1']}")
        print(f"   - ρ = {self.true_params['rho']}")
    
    def demonstrate_individual_differences(self):
        """
        個人差の具体的な例を示す
        """
        print("\n=== 個人差の具体的な例 ===")
        
        if self.data is None:
            print("データが生成されていません。")
            return
        
        # 異なるタイプの釣り人を選ぶ
        angler_stats = self.data.groupby('angler_id').agg({
            'random_intercept': 'first',
            'random_slope': 'first',
            'total_intercept': 'first',
            'total_slope': 'first'
        }).reset_index()
        
        print("\n1. 釣り人の個人差の例:")
        
        # 高能力・高気温効果の釣り人
        high_high = angler_stats.loc[angler_stats['random_intercept'].idxmax()]
        print(f"\n   高能力・高気温効果の釣り人:")
        print(f"   - ランダム切片: {high_high['random_intercept']:.3f}")
        print(f"   - ランダム傾き: {high_high['random_slope']:.3f}")
        print(f"   - 総切片: {high_high['total_intercept']:.3f}")
        print(f"   - 総傾き: {high_high['total_slope']:.3f}")
        
        # 低能力・低気温効果の釣り人
        low_low = angler_stats.loc[angler_stats['random_intercept'].idxmin()]
        print(f"\n   低能力・低気温効果の釣り人:")
        print(f"   - ランダム切片: {low_low['random_intercept']:.3f}")
        print(f"   - ランダム傾き: {low_low['random_slope']:.3f}")
        print(f"   - 総切片: {low_low['total_intercept']:.3f}")
        print(f"   - 総傾き: {low_low['total_slope']:.3f}")
        
        # 平均的な釣り人
        mean_intercept = angler_stats['random_intercept'].mean()
        mean_slope = angler_stats['random_slope'].mean()
        print(f"\n   平均的な釣り人:")
        print(f"   - ランダム切片: {mean_intercept:.3f}")
        print(f"   - ランダム傾き: {mean_slope:.3f}")
        
        print(f"\n2. 気温効果の個人差の解釈:")
        print(f"   - 固定効果: 気温が1度上がると、平均的に釣果尾数が{np.exp(self.true_params['beta_1']):.2f}倍")
        print(f"   - 個人差: 釣り人によって{np.exp(self.true_params['beta_1'] + angler_stats['random_slope'].min()):.2f}倍から{np.exp(self.true_params['beta_1'] + angler_stats['random_slope'].max()):.2f}倍まで変動")
    
    def show_hand_calculation_examples(self):
        """
        手計算による具体例を示す
        """
        print("\n=== 手計算による具体例 ===")
        
        if self.data is None:
            print("データが生成されていません。")
            return
        
        # 特定の釣り人と気温を選んで計算例を示す
        example_data = self.data[(self.data['angler_id'] == 0) & (self.data['trip'] == 1)].iloc[0]
        
        print(f"\n1. 具体例（Angler_1の1回目の釣行）:")
        print(f"   気温: {example_data['temperature']:.1f}度")
        print(f"   ランダム切片: {example_data['random_intercept']:.3f}")
        print(f"   ランダム傾き: {example_data['random_slope']:.3f}")
        
        print(f"\n2. 線形予測子の計算:")
        print(f"   η = β₀ + β₁ × temp + b₀ᵢ + b₁ᵢ × temp")
        print(f"   η = {self.true_params['beta_0']} + {self.true_params['beta_1']} × {example_data['temperature']:.1f} + {example_data['random_intercept']:.3f} + {example_data['random_slope']:.3f} × {example_data['temperature']:.1f}")
        
        linear_pred = (self.true_params['beta_0'] + 
                      self.true_params['beta_1'] * example_data['temperature'] + 
                      example_data['random_intercept'] + 
                      example_data['random_slope'] * example_data['temperature'])
        
        print(f"   η = {linear_pred:.3f}")
        
        print(f"\n3. 期待値の計算:")
        print(f"   λ = exp(η) = exp({linear_pred:.3f})")
        lambda_val = np.exp(linear_pred)
        print(f"   λ = {lambda_val:.3f}")
        
        print(f"\n4. 実際の釣果尾数:")
        print(f"   y ~ Poisson({lambda_val:.3f})")
        print(f"   実際の値: {example_data['catch_count']}")
        
        print(f"\n5. 気温効果の個人差の理解:")
        print(f"   - 固定効果: 気温1度上昇で{np.exp(self.true_params['beta_1']):.2f}倍")
        print(f"   - 個人効果: 気温1度上昇で{np.exp(example_data['total_slope']):.2f}倍")
        print(f"   - 個人差: {np.exp(example_data['total_slope']) / np.exp(self.true_params['beta_1']):.2f}倍の効果")
    
    def visualize_random_slope_model(self):
        """
        ランダム傾きモデルの可視化
        """
        if self.data is None:
            print("データが生成されていません。")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 個別の成長曲線（最初の8人）
        for i in range(min(8, self.data['angler_id'].nunique())):
            angler_data = self.data[self.data['angler_id'] == i]
            axes[0,0].plot(angler_data['temperature'], angler_data['catch_count'], 
                          marker='o', alpha=0.7, label=f'Angler_{i+1}')
        
        axes[0,0].set_title('個別の成長曲線（最初の8人）')
        axes[0,0].set_xlabel('気温（度）')
        axes[0,0].set_ylabel('釣果尾数')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. ランダム切片の分布
        angler_effects = self.data.groupby('angler_id')['random_intercept'].first()
        axes[0,1].hist(angler_effects, bins=10, alpha=0.7, edgecolor='black')
        axes[0,1].axvline(0, color='red', linestyle='--', label='平均=0')
        axes[0,1].set_title('ランダム切片の分布')
        axes[0,1].set_xlabel('ランダム切片')
        axes[0,1].set_ylabel('頻度')
        axes[0,1].legend()
        
        # 3. ランダム傾きの分布
        angler_slopes = self.data.groupby('angler_id')['random_slope'].first()
        axes[0,2].hist(angler_slopes, bins=10, alpha=0.7, edgecolor='black')
        axes[0,2].axvline(0, color='red', linestyle='--', label='平均=0')
        axes[0,2].set_title('ランダム傾きの分布')
        axes[0,2].set_xlabel('ランダム傾き')
        axes[0,2].set_ylabel('頻度')
        axes[0,2].legend()
        
        # 4. ランダム切片とランダム傾きの散布図
        axes[1,0].scatter(angler_effects, angler_slopes, alpha=0.7)
        axes[1,0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1,0].set_title('ランダム切片とランダム傾きの関係')
        axes[1,0].set_xlabel('ランダム切片')
        axes[1,0].set_ylabel('ランダム傾き')
        
        # 5. 気温別の平均釣果尾数（全体）
        temp_means = self.data.groupby('temperature')['catch_count'].agg(['mean', 'std']).reset_index()
        axes[1,1].plot(temp_means['temperature'], temp_means['mean'], 'ro-', linewidth=2)
        axes[1,1].fill_between(temp_means['temperature'], 
                              temp_means['mean'] - temp_means['std'],
                              temp_means['mean'] + temp_means['std'], 
                              alpha=0.3)
        axes[1,1].set_title('気温別の平均釣果尾数（全体）')
        axes[1,1].set_xlabel('気温（度）')
        axes[1,1].set_ylabel('平均釣果尾数')
        
        # 6. 個人別の気温効果
        angler_slopes_total = self.data.groupby('angler_id')['total_slope'].first()
        axes[1,2].hist(angler_slopes_total, bins=10, alpha=0.7, edgecolor='black')
        axes[1,2].axvline(self.true_params['beta_1'], color='red', linestyle='--', 
                          label=f'固定効果={self.true_params["beta_1"]}')
        axes[1,2].set_title('個人別の気温効果（総傾き）の分布')
        axes[1,2].set_xlabel('総傾き（固定効果+ランダム効果）')
        axes[1,2].set_ylabel('頻度')
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """
        異なるモデルの比較
        """
        print("\n=== モデルの比較 ===")
        
        print("\n1. 単純なポアソン回帰（ランダム効果なし）:")
        print("   log(λ_ij) = β₀ + β₁ × temp_ij")
        print("   問題: 個人差を完全に無視")
        
        print("\n2. ランダム切片モデル（従来のGLMM）:")
        print("   log(λ_ij) = β₀ + β₁ × temp_ij + b₀ᵢ")
        print("   問題: 気温効果の個人差を無視")
        
        print("\n3. ランダム傾きモデル（今回のモデル）:")
        print("   log(λ_ij) = β₀ + β₁ × temp_ij + b₀ᵢ + b₁ᵢ × temp_ij")
        print("   利点: 基本能力と気温効果の両方の個人差を考慮")
        
        print("\n4. モデルの複雑さと解釈性:")
        print("   - 単純なモデル: 解釈しやすいが、現実を反映しない")
        print("   - 複雑なモデル: 現実を反映するが、解釈が難しい")
        print("   - バランスが重要")
    
    def demonstrate_likelihood_structure(self):
        """
        尤度関数の構造を説明
        """
        print("\n=== 尤度関数の構造 ===")
        
        print("\n1. 条件付き尤度（ランダム効果が与えられた場合）:")
        print("   f(y_i | b₀ᵢ, b₁ᵢ) = ∏ᵢ Poisson(y_ij | λ_ij)")
        print("   ここで λ_ij = exp(β₀ + β₁ × temp_ij + b₀ᵢ + b₁ᵢ × temp_ij)")
        
        print("\n2. ランダム効果の事前分布:")
        print("   f(b₀ᵢ, b₁ᵢ) = N((b₀ᵢ, b₁ᵢ) | 0, G)")
        print("   ここで G は2×2の分散共分散行列")
        
        print("\n3. 完全な尤度:")
        print("   f(y_i, b₀ᵢ, b₁ᵢ) = f(y_i | b₀ᵢ, b₁ᵢ) × f(b₀ᵢ, b₁ᵢ)")
        
        print("\n4. 全データの尤度:")
        print("   f(y, b) = ∏ᵢ f(y_i, b₀ᵢ, b₁ᵢ)")
        
        print("\n5. 積分尤度:")
        print("   f(y) = ∫∫ f(y, b₀, b₁) db₀ db₁")
        print("   この二重積分を最大化することでパラメータを推定")
        
        print("\n6. 推定の複雑さ:")
        print("   - ランダム切片モデル: 1次元積分")
        print("   - ランダム傾きモデル: 2次元積分")
        print("   - 計算コストが高くなる")

def main():
    """
    メイン関数
    """
    print("🎣 釣果尾数のランダム傾きGLMM例")
    print("=" * 60)
    
    # 例のインスタンスを作成
    example = FishingRandomSlopeGLMM()
    
    # サンプルデータの作成
    print("\n1. サンプルデータの作成...")
    data = example.create_random_slope_data()
    print("データ作成完了！")
    
    # ランダム傾きモデルの構造説明
    print("\n2. ランダム傾きモデルの構造説明...")
    example.explain_random_slope_model()
    
    # 個人差の具体的な例
    print("\n3. 個人差の具体的な例...")
    example.demonstrate_individual_differences()
    
    # 手計算による具体例
    print("\n4. 手計算による具体例...")
    example.show_hand_calculation_examples()
    
    # データの可視化
    print("\n5. データの可視化...")
    example.visualize_random_slope_model()
    
    # モデルの比較
    print("\n6. モデルの比較...")
    example.compare_models()
    
    # 尤度関数の構造
    print("\n7. 尤度関数の構造...")
    example.demonstrate_likelihood_structure()
    
    print("\n" + "=" * 60)
    print("説明完了！")
    print("ランダム傾きモデルについて理解が深まりましたか？")

if __name__ == "__main__":
    main()