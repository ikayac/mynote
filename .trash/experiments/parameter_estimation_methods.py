#!/usr/bin/env python3
"""
ランダム効果のパラメータ推定方法

このファイルでは、直接観測できないランダム効果を
どのようにパラメータ推定するかを詳しく説明します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class ParameterEstimationMethods:
    """
    ランダム効果のパラメータ推定方法を詳しく説明するクラス
    """
    
    def __init__(self):
        self.data = None
        self.true_params = {}
        
    def create_simple_data(self, n_subjects=15, n_obs_per_subject=8):
        """
        シンプルなサンプルデータを作成
        
        Parameters:
        -----------
        n_subjects : int
            被験者数
        n_obs_per_subject : int
            各被験者の観測数
        """
        np.random.seed(42)
        
        # 真のパラメータ
        self.true_params = {
            'beta_0': 2.0,      # 固定切片
            'beta_1': 0.5,      # 固定傾き
            'sigma_b': 1.2      # ランダム効果の標準偏差
        }
        
        # ランダム効果の生成
        random_effects = np.random.normal(0, self.true_params['sigma_b'], n_subjects)
        
        # 説明変数（時間）
        time_range = np.linspace(0, 7, n_obs_per_subject)
        
        data_list = []
        
        for i in range(n_subjects):
            b_i = random_effects[i]
            
            for j, time in enumerate(time_range):
                # 線形予測子
                linear_predictor = (self.true_params['beta_0'] + 
                                  self.true_params['beta_1'] * time + 
                                  b_i)
                
                # 期待値（指数関数で変換）
                lambda_param = np.exp(linear_predictor)
                
                # ポアソン分布からのサンプリング
                y = np.random.poisson(lambda_param)
                
                data_list.append({
                    'subject_id': i,
                    'subject': f'Subject_{i+1}',
                    'time': time,
                    'y': y,
                    'lambda_param': lambda_param,
                    'linear_predictor': linear_predictor,
                    'true_random_effect': b_i
                })
        
        self.data = pd.DataFrame(data_list)
        return self.data
    
    def explain_estimation_problem(self):
        """
        推定問題の本質を説明
        """
        print("=== ランダム効果の推定問題 ===")
        
        print("\n1. 問題の本質:")
        print("   - ランダム効果 b_i は直接観測できない")
        print("   - 観測できるのは応答変数 y_ij と説明変数 x_ij のみ")
        print("   - しかし、推定したいのは固定効果β、分散σ²_b、個別のb_i")
        
        print("\n2. 数学的な困難:")
        print("   完全な尤度関数:")
        print("   f(y) = ∫ f(y | b) × f(b) db")
        print("   この積分を解析的に解くことは困難")
        
        print("\n3. 解決方法:")
        print("   - 最尤推定: ラプラス近似、ガウス・エルミート求積")
        print("   - ベイズ推定: MCMC法")
        print("   - その他: EMアルゴリズム、変分推論")
    
    def demonstrate_laplace_approximation(self):
        """
        ラプラス近似法を詳しく説明
        """
        print("\n=== ラプラス近似法による最尤推定 ===")
        
        print("\n1. ラプラス近似の基本的な考え方:")
        print("   - ランダム効果の最尤推定値を見つける")
        print("   - その周りでテイラー展開")
        print("   - 積分を近似")
        
        print("\n2. 数学的定式化:")
        print("   対数尤度関数:")
        print("   ℓ(β, σ²_b) = log ∫ f(y | b) × f(b) db")
        print("   ")
        print("   ラプラス近似:")
        print("   ℓ(β, σ²_b) ≈ log f(y | b̂) + log f(b̂) - (1/2) log |H|")
        print("   ")
        print("   ここで:")
        print("   - b̂: ランダム効果の最尤推定値")
        print("   - H: ヘッセ行列（2階微分）")
        
        print("\n3. 実装の手順:")
        print("   Step 1: 固定効果βと分散σ²_bを初期化")
        print("   Step 2: ランダム効果b_iを推定（条件付き最尤）")
        print("   Step 3: ヘッセ行列Hを計算")
        print("   Step 4: ラプラス近似による対数尤度を計算")
        print("   Step 5: 対数尤度を最大化するβとσ²_bを更新")
        print("   Step 6: Step 2-5を収束するまで繰り返し")
        
        # 実際の実装例
        self._implement_laplace_approximation()
    
    def _implement_laplace_approximation(self):
        """
        ラプラス近似の実装例
        """
        print("\n4. 実装例（簡略版）:")
        
        if self.data is None:
            print("データが生成されていません。")
            return
        
        # データの準備
        subjects = self.data['subject_id'].unique()
        n_subjects = len(subjects)
        
        print(f"\n   データ概要:")
        print(f"   - 被験者数: {n_subjects}")
        print(f"   - 総観測数: {len(self.data)}")
        
        # 初期パラメータ
        beta_0_init = 1.5
        beta_1_init = 0.3
        sigma_b_init = 1.0
        
        print(f"\n   初期パラメータ:")
        print(f"   - β₀ = {beta_0_init}")
        print(f"   - β₁ = {beta_1_init}")
        print(f"   - σ_b = {sigma_b_init}")
        
        # ランダム効果の推定（簡略版）
        print(f"\n   ランダム効果の推定（簡略版）:")
        print(f"   各被験者について、条件付き最尤推定を実行")
        
        # 実際の推定は複雑なので、概念的な説明
        print(f"\n   実際の推定では:")
        print(f"   1. 各被験者のデータから個別のランダム効果を推定")
        print(f"   2. ヘッセ行列を計算")
        print(f"   3. ラプラス近似による対数尤度を計算")
        print(f"   4. パラメータを更新")
        print(f"   5. 収束するまで繰り返し")
        
        print(f"\n   注意: 実際の実装は非常に複雑で、")
        print(f"   通常は専用の統計パッケージ（lme4、statsmodels等）を使用")
    
    def demonstrate_mcmc_method(self):
        """
        MCMC法によるベイズ推定を詳しく説明
        """
        print("\n=== MCMC法によるベイズ推定 ===")
        
        print("\n1. MCMC法の基本的な考え方:")
        print("   - ランダム効果を隠れ変数として扱う")
        print("   - 事後分布からのサンプリングを行う")
        print("   - パラメータとランダム効果を同時に推定")
        
        print("\n2. ベイズモデルの構造:")
        print("   事前分布:")
        print("   p(β₀) ~ N(0, 100)")
        print("   p(β₁) ~ N(0, 100)")
        print("   p(σ²_b) ~ InvGamma(0.01, 0.01)")
        print("   ")
        print("   ランダム効果の事前分布:")
        print("   p(b_i | σ²_b) ~ N(0, σ²_b)")
        print("   ")
        print("   尤度:")
        print("   p(y_ij | β₀, β₁, b_i) ~ Poisson(exp(β₀ + β₁ × time_ij + b_i))")
        
        print("\n3. 事後分布:")
        print("   p(β₀, β₁, σ²_b, b | y) ∝ p(y | β₀, β₁, b) × p(b | σ²_b) × p(β₀) × p(β₁) × p(σ²_b)")
        
        print("\n4. MCMCアルゴリズム（ギブスサンプラー）:")
        print("   Step 1: パラメータを初期化")
        print("   Step 2: ランダム効果b_iを更新（条件付き事後分布から）")
        print("   Step 3: 固定効果β₀, β₁を更新（条件付き事後分布から）")
        print("   Step 4: 分散σ²_bを更新（条件付き事後分布から）")
        print("   Step 5: Step 2-4を指定回数繰り返し")
        
        # 実際の実装例
        self._implement_mcmc_example()
    
    def _implement_mcmc_example(self):
        """
        MCMC法の実装例（簡略版）
        """
        print("\n5. 実装例（簡略版）:")
        
        if self.data is None:
            print("データが生成されていません。")
            return
        
        print(f"\n   ギブスサンプラーの各ステップ:")
        
        # Step 1: ランダム効果の更新
        print(f"\n   Step 1: ランダム効果b_iの更新")
        print(f"   条件付き事後分布:")
        print(f"   b_i | y_i, β₀, β₁, σ²_b ~ N(μ_b, σ²_b_post)")
        print(f"   ")
        print(f"   ここで:")
        print(f"   μ_b = (Σ_j y_ij - exp(β₀ + β₁ × time_ij)) / (n_i + 1/σ²_b)")
        print(f"   σ²_b_post = 1 / (n_i + 1/σ²_b)")
        print(f"   n_i: 被験者iの観測数")
        
        # Step 2: 固定効果の更新
        print(f"\n   Step 2: 固定効果β₀, β₁の更新")
        print(f"   条件付き事後分布:")
        print(f"   β₀, β₁ | y, b, σ²_b ~ N(μ_β, Σ_β)")
        print(f"   ")
        print(f"   ここで:")
        print(f"   μ_β: 線形回帰の最尤推定値")
        print(f"   Σ_β: 線形回帰の共分散行列")
        
        # Step 3: 分散の更新
        print(f"\n   Step 3: 分散σ²_bの更新")
        print(f"   条件付き事後分布:")
        print(f"   σ²_b | b ~ InvGamma(α_post, β_post)")
        print(f"   ")
        print(f"   ここで:")
        print(f"   α_post = α_prior + n/2")
        print(f"   β_post = β_prior + Σ_i b_i²/2")
        print(f"   n: 被験者数")
        
        print(f"\n   注意: 実際の実装は非常に複雑で、")
        print(f"   通常は専用のベイズ統計パッケージ（PyMC、Stan等）を使用")
    
    def compare_estimation_methods(self):
        """
        推定方法の比較
        """
        print("\n=== 推定方法の比較 ===")
        
        print("\n1. 最尤推定（ラプラス近似）:")
        print("   利点:")
        print("   - 計算が比較的高速")
        print("   - 標準誤差が得られる")
        print("   - 解釈が比較的容易")
        print("   ")
        print("   欠点:")
        print("   - 近似に依存")
        print("   - 小サンプルでの偏り")
        print("   - 複雑なモデルでは収束しない場合がある")
        
        print("\n2. MCMC法（ベイズ推定）:")
        print("   利点:")
        print("   - 正確な事後分布が得られる")
        print("   - 不確実性の適切な評価")
        print("   - 事前知識の活用")
        print("   - 複雑なモデルでも適用可能")
        print("   ")
        print("   欠点:")
        print("   - 計算時間が長い")
        print("   - 収束診断が必要")
        print("   - 事前分布の選択が重要")
        print("   - 解釈が複雑")
        
        print("\n3. 使い分け:")
        print("   - 探索的分析: 最尤推定")
        print("   - 最終的な推論: MCMC法")
        print("   - 大規模データ: 最尤推定")
        print("   - 小サンプル: MCMC法")
    
    def show_practical_implementation(self):
        """
        実用的な実装方法を説明
        """
        print("\n=== 実用的な実装方法 ===")
        
        print("\n1. R言語での実装:")
        print("   最尤推定:")
        print("   library(lme4)")
        print("   model <- glmer(y ~ time + (1|subject), family=poisson, data=data)")
        print("   ")
        print("   MCMC法:")
        print("   library(rstanarm)")
        print("   model <- stan_glmer(y ~ time + (1|subject), family=poisson, data=data)")
        
        print("\n2. Pythonでの実装:")
        print("   最尤推定:")
        print("   from statsmodels.regression.mixed_linear_model import MixedLM")
        print("   model = MixedLM(endog=y, exog=X, groups=subject)")
        print("   ")
        print("   MCMC法:")
        print("   import pymc as pm")
        print("   with pm.Model() as model:")
        print("       # モデルの定義")
        print("       # MCMCサンプリング")
        
        print("\n3. 実装の注意点:")
        print("   - データの前処理が重要")
        print("   - 初期値の設定")
        print("   - 収束性の確認")
        print("   - モデルの診断")
        print("   - 結果の解釈")
    
    def demonstrate_convergence_diagnostics(self):
        """
        収束診断の方法を説明
        """
        print("\n=== 収束診断の方法 ===")
        
        print("\n1. 最尤推定の収束診断:")
        print("   - 対数尤度の収束")
        print("   - パラメータ推定値の安定性")
        print("   - 標準誤差の妥当性")
        print("   - ヘッセ行列の正定値性")
        
        print("\n2. MCMC法の収束診断:")
        print("   - トレースプロット")
        print("   - Gelman-Rubin統計量")
        print("   - 有効サンプルサイズ")
        print("   - 自己相関の確認")
        
        print("\n3. モデルの妥当性:")
        print("   - 残差分析")
        print("   - 影響度分析")
        print("   - 予測の妥当性")
        print("   - ランダム効果の分布の妥当性")
        
        print("\n4. 実用的なチェックリスト:")
        print("   □ 対数尤度が収束しているか")
        print("   □ パラメータ推定値が安定しているか")
        print("   □ 標準誤差が妥当か")
        print("   □ 残差が適切な分布をしているか")
        print("   □ ランダム効果が正規分布に従っているか")

def main():
    """
    メイン関数
    """
    print("🔍 ランダム効果のパラメータ推定方法")
    print("=" * 60)
    
    # 例のインスタンスを作成
    example = ParameterEstimationMethods()
    
    # サンプルデータの作成
    print("\n1. サンプルデータの作成...")
    data = example.create_simple_data()
    print("データ作成完了！")
    
    # 推定問題の本質
    print("\n2. 推定問題の本質...")
    example.explain_estimation_problem()
    
    # ラプラス近似法
    print("\n3. ラプラス近似法による最尤推定...")
    example.demonstrate_laplace_approximation()
    
    # MCMC法
    print("\n4. MCMC法によるベイズ推定...")
    example.demonstrate_mcmc_method()
    
    # 推定方法の比較
    print("\n5. 推定方法の比較...")
    example.compare_estimation_methods()
    
    # 実用的な実装方法
    print("\n6. 実用的な実装方法...")
    example.show_practical_implementation()
    
    # 収束診断
    print("\n7. 収束診断の方法...")
    example.demonstrate_convergence_diagnostics()
    
    print("\n" + "=" * 60)
    print("説明完了！")
    print("ランダム効果のパラメータ推定について理解が深まりましたか？")

if __name__ == "__main__":
    main()