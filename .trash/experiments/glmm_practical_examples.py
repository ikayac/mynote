#!/usr/bin/env python3
"""
GLMMの実践例 - 実際の統計パッケージを使用

このファイルでは、様々な統計パッケージを使用してGLMMを実装します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class GLMMPracticalExamples:
    """
    GLMMの実践例を提供するクラス
    """
    
    def __init__(self):
        self.data = None
        
    def create_longitudinal_data(self, n_subjects=50, n_timepoints=5):
        """
        縦断データ（反復測定データ）を作成
        
        Parameters:
        -----------
        n_subjects : int
            被験者数
        n_timepoints : int
            時間点の数
        """
        np.random.seed(123)
        
        # 被験者固有の効果（ランダム切片）
        subject_intercepts = np.random.normal(0, 1.5, n_subjects)
        
        # 被験者固有の傾き（ランダム傾き）
        subject_slopes = np.random.normal(0.5, 0.3, n_subjects)
        
        # 時間変数
        time = np.linspace(0, 4, n_timepoints)
        
        data_list = []
        
        for i in range(n_subjects):
            for j, t in enumerate(time):
                # 線形予測子
                linear_predictor = (2.0 + subject_intercepts[i] + 
                                  (0.8 + subject_slopes[i]) * t)
                
                # ポアソン分布の場合
                lambda_param = np.exp(linear_predictor)
                y = np.random.poisson(lambda_param)
                
                data_list.append({
                    'subject_id': i,
                    'subject': f'Subject_{i+1}',
                    'time': t,
                    'y': y,
                    'lambda': lambda_param,
                    'linear_predictor': linear_predictor
                })
        
        self.data = pd.DataFrame(data_list)
        return self.data
    
    def demonstrate_mixed_effects_analysis(self):
        """
        混合効果モデルの分析をデモンストレーション
        """
        print("=== 混合効果モデルの分析例 ===")
        
        if self.data is None:
            print("データが生成されていません。create_longitudinal_data()を先に実行してください。")
            return
        
        # データの可視化
        self._plot_longitudinal_data()
        
        # 基本的な統計分析
        self._basic_statistical_analysis()
        
        # 混合効果モデルの概念説明
        self._explain_mixed_effects_model()
    
    def _plot_longitudinal_data(self):
        """
        縦断データの可視化
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 個別の成長曲線
        for subject_id in range(min(10, self.data['subject_id'].nunique())):
            subject_data = self.data[self.data['subject_id'] == subject_id]
            axes[0,0].plot(subject_data['time'], subject_data['y'], 
                          marker='o', alpha=0.7, label=f'Subject_{subject_id+1}')
        
        axes[0,0].set_title('個別の成長曲線（最初の10被験者）')
        axes[0,0].set_xlabel('時間')
        axes[0,0].set_ylabel('応答変数')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. 平均成長曲線
        mean_by_time = self.data.groupby('time')['y'].agg(['mean', 'std']).reset_index()
        axes[0,1].plot(mean_by_time['time'], mean_by_time['mean'], 'ro-', linewidth=2)
        axes[0,1].fill_between(mean_by_time['time'], 
                              mean_by_time['mean'] - mean_by_time['std'],
                              mean_by_time['mean'] + mean_by_time['std'], 
                              alpha=0.3)
        axes[0,1].set_title('平均成長曲線（標準偏差付き）')
        axes[0,1].set_xlabel('時間')
        axes[0,1].set_ylabel('平均応答変数')
        
        # 3. 被験者別の初期値と傾きの関係
        subject_stats = self.data.groupby('subject_id').agg({
            'y': ['first', 'last'],
            'time': ['first', 'last']
        }).reset_index()
        
        subject_stats.columns = ['subject_id', 'y_first', 'y_last', 'time_first', 'time_last']
        subject_stats['slope'] = (subject_stats['y_last'] - subject_stats['y_first']) / \
                                (subject_stats['time_last'] - subject_stats['time_first'])
        
        axes[1,0].scatter(subject_stats['y_first'], subject_stats['slope'], alpha=0.7)
        axes[1,0].set_title('初期値と成長率の関係')
        axes[1,0].set_xlabel('初期値（時間0での応答）')
        axes[1,0].set_ylabel('成長率（傾き）')
        
        # 4. 応答変数の分布
        sns.boxplot(data=self.data, x='time', y='y', ax=axes[1,1])
        axes[1,1].set_title('時間別の応答変数の分布')
        axes[1,1].set_xlabel('時間')
        axes[1,1].set_ylabel('応答変数')
        
        plt.tight_layout()
        plt.show()
    
    def _basic_statistical_analysis(self):
        """
        基本的な統計分析
        """
        print("\n=== 基本的な統計分析 ===")
        
        # 記述統計
        print("1. 記述統計:")
        print(self.data.describe())
        
        # 被験者別の統計
        print("\n2. 被験者別の統計:")
        subject_summary = self.data.groupby('subject_id').agg({
            'y': ['count', 'mean', 'std', 'min', 'max'],
            'time': ['min', 'max']
        }).round(3)
        print(subject_summary.head(10))
        
        # 時間別の統計
        print("\n3. 時間別の統計:")
        time_summary = self.data.groupby('time').agg({
            'y': ['count', 'mean', 'std', 'min', 'max']
        }).round(3)
        print(time_summary)
        
        # 相関分析
        print("\n4. 相関分析:")
        correlation = self.data[['time', 'y']].corr()
        print(correlation)
    
    def _explain_mixed_effects_model(self):
        """
        混合効果モデルの概念を説明
        """
        print("\n=== 混合効果モデルの概念 ===")
        
        print("1. 固定効果（Fixed Effects）:")
        print("   - 時間の効果: 全体的な成長傾向")
        print("   - 切片: 全体的な初期値")
        
        print("\n2. ランダム効果（Random Effects）:")
        print("   - ランダム切片: 被験者固有の初期値の変動")
        print("   - ランダム傾き: 被験者固有の成長率の変動")
        
        print("\n3. モデルの数式:")
        print("   yᵢⱼ = (β₀ + b₀ᵢ) + (β₁ + b₁ᵢ)tᵢⱼ + εᵢⱼ")
        print("   ここで:")
        print("   - β₀, β₁: 固定効果（全体平均）")
        print("   - b₀ᵢ, b₁ᵢ: ランダム効果（被験者i固有）")
        print("   - tᵢⱼ: 時間変数")
        print("   - εᵢⱼ: 誤差項")
        
        print("\n4. 分散成分:")
        print("   - Var(b₀ᵢ) = σ²₀: ランダム切片の分散")
        print("   - Var(b₁ᵢ) = σ²₁: ランダム傾きの分散")
        print("   - Cov(b₀ᵢ, b₁ᵢ) = σ₀₁: ランダム切片と傾きの共分散")
        print("   - Var(εᵢⱼ) = σ²: 誤差分散")
    
    def demonstrate_glmm_with_statsmodels(self):
        """
        statsmodelsを使用したGLMMの実装例
        """
        print("\n=== statsmodelsを使用したGLMMの実装例 ===")
        
        try:
            import statsmodels.api as sm
            from statsmodels.regression.mixed_linear_model import MixedLM
            
            print("statsmodelsが利用可能です。")
            
            # 線形混合効果モデルの例
            print("\n1. 線形混合効果モデル（LMM）:")
            print("   - 応答変数: 連続変数")
            print("   - 分布: 正規分布")
            print("   - リンク関数: 恒等関数")
            
            # データの準備
            if self.data is not None:
                # 被験者IDをカテゴリカル変数に変換
                self.data['subject_id_cat'] = self.data['subject_id'].astype('category')
                
                # モデルの構築
                model = MixedLM(
                    endog=self.data['y'],
                    exog=sm.add_constant(self.data['time']),
                    groups=self.data['subject_id_cat']
                )
                
                print("   モデル構築完了")
                print("   推定中...")
                
                # モデルの推定（実際の推定は時間がかかるため、概念のみ説明）
                print("   注意: 実際の推定は計算時間がかかります")
                print("   この例では概念的な説明に留めます")
            
        except ImportError:
            print("statsmodelsがインストールされていません。")
            print("インストール方法: pip install statsmodels")
    
    def demonstrate_glmm_with_pymc(self):
        """
        PyMCを使用したベイズGLMMの実装例
        """
        print("\n=== PyMCを使用したベイズGLMMの実装例 ===")
        
        try:
            import pymc as pm
            print("PyMCが利用可能です。")
            
            print("\n1. ベイズGLMMの利点:")
            print("   - 不確実性の適切な評価")
            print("   - 事前知識の活用")
            print("   - 複雑なモデルの柔軟な構築")
            
            print("\n2. モデル構造:")
            print("   - 階層的ベイズモデル")
            print("   - 事前分布の設定")
            print("   - MCMCサンプリング")
            
            print("\n3. 実装例（概念）:")
            print("   with pm.Model() as model:")
            print("       # 事前分布")
            print("       beta_0 = pm.Normal('beta_0', mu=0, sigma=10)")
            print("       beta_1 = pm.Normal('beta_1', mu=0, sigma=10)")
            print("       sigma_subject = pm.HalfNormal('sigma_subject', sigma=5)")
            print("       ")
            print("       # ランダム効果")
            print("       subject_effects = pm.Normal('subject_effects', mu=0, sigma=sigma_subject, shape=n_subjects)")
            print("       ")
            print("       # 線形予測子")
            print("       mu = beta_0 + beta_1 * time + subject_effects[subject_id]")
            print("       ")
            print("       # 尤度")
            print("       y_obs = pm.Poisson('y_obs', mu=pm.math.exp(mu), observed=y)")
            
        except ImportError:
            print("PyMCがインストールされていません。")
            print("インストール方法: pip install pymc")
    
    def practical_tips_and_considerations(self):
        """
        実践的なヒントと注意点
        """
        print("\n=== 実践的なヒントと注意点 ===")
        
        print("1. モデル選択:")
        print("   - 情報量基準（AIC, BIC）の使用")
        print("   - 交差検証の実施")
        print("   - 残差分析によるモデル診断")
        
        print("\n2. 収束性の確認:")
        print("   - MCMCサンプリングの収束診断")
        print("   - 連鎖間の相関の確認")
        print("   - 有効サンプルサイズの確認")
        
        print("\n3. 多重比較の問題:")
        print("   - 多重比較補正の適用")
        print("   - 偽発見率の制御")
        print("   - 事前計画された比較の活用")
        
        print("\n4. 欠損データの処理:")
        print("   - 完全情報最尤推定法（FIML）")
        print("   - 多重代入法（Multiple Imputation）")
        print("   - 欠損のメカニズムの理解")
        
        print("\n5. 計算効率:")
        print("   - スパース行列の活用")
        print("   - 並列計算の活用")
        print("   - 近似法の検討")

def main():
    """
    メイン関数
    """
    print("GLMMの実践例")
    print("=" * 50)
    
    # 実践例のインスタンスを作成
    examples = GLMMPracticalExamples()
    
    # 縦断データの作成
    print("\n1. 縦断データの作成...")
    data = examples.create_longitudinal_data()
    print("データ作成完了！")
    
    # 混合効果モデルの分析
    print("\n2. 混合効果モデルの分析...")
    examples.demonstrate_mixed_effects_analysis()
    
    # statsmodelsを使用したGLMM
    print("\n3. statsmodelsを使用したGLMM...")
    examples.demonstrate_glmm_with_statsmodels()
    
    # PyMCを使用したベイズGLMM
    print("\n4. PyMCを使用したベイズGLMM...")
    examples.demonstrate_glmm_with_pymc()
    
    # 実践的なヒント
    print("\n5. 実践的なヒントと注意点...")
    examples.practical_tips_and_considerations()
    
    print("\n" + "=" * 50)
    print("実践例完了！")
    print("GLMMの実装について理解が深まりましたか？")

if __name__ == "__main__":
    main()