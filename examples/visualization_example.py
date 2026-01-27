#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可視化の例を示すサンプルスクリプト
Google Colabで実行することを想定
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def generate_sample_data():
    """サンプルデータの生成"""
    np.random.seed(42)
    
    # 混合切断正規分布のパラメータ
    mu1, sigma1 = 270, 15  # 第1ピーク（270付近）
    mu2, sigma2 = 300, 12  # 第2ピーク（300付近）
    weights = [0.6, 0.4]   # 混合比率
    truncation_point = 235  # 切断点
    
    # 各成分からサンプリング
    n1 = int(135 * weights[0])
    n2 = 135 - n1
    
    # 第1成分（270付近）
    samples1 = np.random.normal(mu1, sigma1, n1 * 3)
    samples1 = samples1[samples1 >= truncation_point][:n1]
    
    # 第2成分（300付近）
    samples2 = np.random.normal(mu2, sigma2, n2 * 3)
    samples2 = samples2[samples2 >= truncation_point][:n2]
    
    # 結合
    data = np.concatenate([samples1, samples2])
    np.random.shuffle(data)
    
    return data

def create_basic_visualization(data):
    """基本的な可視化の作成"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. ヒストグラム
    axes[0, 0].hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].set_title('強度のヒストグラム', fontsize=14)
    axes[0, 0].set_xlabel('強度 (MPa)')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 箱ひげ図
    axes[0, 1].boxplot(data, vert=True)
    axes[0, 1].set_title('強度の箱ひげ図', fontsize=14)
    axes[0, 1].set_ylabel('強度 (MPa)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Qプロット（正規分布との比較）
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Qプロット（正規分布との比較）', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 累積分布関数
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1, 1].plot(sorted_data, y, 'b-', linewidth=2, label='経験的CDF')
    
    # 理論的正規分布のCDF
    mu, sigma = np.mean(data), np.std(data)
    x_theory = np.linspace(min(data), max(data), 100)
    cdf_theory = stats.norm.cdf(x_theory, mu, sigma)
    axes[1, 1].plot(x_theory, cdf_theory, 'r--', linewidth=2, label='理論的正規分布CDF')
    
    axes[1, 1].set_title('累積分布関数の比較', fontsize=14)
    axes[1, 1].set_xlabel('強度 (MPa)')
    axes[1, 1].set_ylabel('累積確率')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basic_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_distribution_comparison(data):
    """分布比較の可視化"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 密度曲線用のx軸
    x_range = np.linspace(235, max(data) + 25, 200)
    
    # 1. 正規分布
    mu, sigma = np.mean(data), np.std(data)
    axes[0, 0].hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2, label='正規分布')
    axes[0, 0].set_title('正規分布との比較', fontsize=12)
    axes[0, 0].set_xlabel('強度 (MPa)')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ワイブル分布
    try:
        shape, loc, scale = stats.weibull_min.fit(data)
        axes[0, 1].hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 1].plot(x_range, stats.weibull_min.pdf(x_range, shape, loc, scale), 'g-', linewidth=2, label='ワイブル分布')
        axes[0, 1].set_title('ワイブル分布との比較', fontsize=12)
        axes[0, 1].set_xlabel('強度 (MPa)')
        axes[0, 1].set_ylabel('密度')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    except:
        axes[0, 1].text(0.5, 0.5, 'ワイブル分布\nフィッティング失敗', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('ワイブル分布との比較', fontsize=12)
    
    # 3. 対数正規分布
    try:
        shape, loc, scale = stats.lognorm.fit(data)
        axes[0, 2].hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 2].plot(x_range, stats.lognorm.pdf(x_range, shape, loc, scale), 'b-', linewidth=2, label='対数正規分布')
        axes[0, 2].set_title('対数正規分布との比較', fontsize=12)
        axes[0, 2].set_xlabel('強度 (MPa)')
        axes[0, 2].set_ylabel('密度')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    except:
        axes[0, 2].text(0.5, 0.5, '対数正規分布\nフィッティング失敗', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('対数正規分布との比較', fontsize=12)
    
    # 4. ガンマ分布
    try:
        shape, loc, scale = stats.gamma.fit(data)
        axes[1, 0].hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 0].plot(x_range, stats.gamma.pdf(x_range, shape, loc, scale), 'm-', linewidth=2, label='ガンマ分布')
        axes[1, 0].set_title('ガンマ分布との比較', fontsize=12)
        axes[1, 0].set_xlabel('強度 (MPa)')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    except:
        axes[1, 0].text(0.5, 0.5, 'ガンマ分布\nフィッティング失敗', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('ガンマ分布との比較', fontsize=12)
    
    # 5. 切断正規分布
    truncation_point = 235
    mask = x_range >= truncation_point
    if np.any(mask):
        norm_factor = 1 - stats.norm.cdf(truncation_point, mu, sigma)
        if norm_factor > 1e-10:
            density = stats.norm.pdf(x_range[mask], mu, sigma) / norm_factor
            axes[1, 1].hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
            axes[1, 1].plot(x_range[mask], density, 'c-', linewidth=2, label='切断正規分布')
            axes[1, 1].set_title('切断正規分布との比較', fontsize=12)
            axes[1, 1].set_xlabel('強度 (MPa)')
            axes[1, 1].set_ylabel('密度')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 混合正規分布
    try:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data.reshape(-1, 1))
        
        density = np.zeros_like(x_range)
        for i in range(2):
            density += gmm.weights_[i] * stats.norm.pdf(x_range, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0]))
        
        axes[1, 2].hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 2].plot(x_range, density, 'orange', linewidth=2, label='混合正規分布')
        axes[1, 2].set_title('混合正規分布との比較', fontsize=12)
        axes[1, 2].set_xlabel('強度 (MPa)')
        axes[1, 2].set_ylabel('密度')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    except:
        axes[1, 2].text(0.5, 0.5, '混合正規分布\nフィッティング失敗', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('混合正規分布との比較', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_statistical_summary(data):
    """統計的サマリーの可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 基本統計量
    stats_data = {
        '平均': np.mean(data),
        '中央値': np.median(data),
        '標準偏差': np.std(data),
        '最小値': np.min(data),
        '最大値': np.max(data),
        '歪度': stats.skew(data),
        '尖度': stats.kurtosis(data)
    }
    
    axes[0, 0].bar(range(len(stats_data)), list(stats_data.values()), color='skyblue')
    axes[0, 0].set_title('基本統計量', fontsize=14)
    axes[0, 0].set_xticks(range(len(stats_data)))
    axes[0, 0].set_xticklabels(list(stats_data.keys()), rotation=45, ha='right')
    axes[0, 0].set_ylabel('値')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 分位点
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = [np.percentile(data, p) for p in percentiles]
    
    axes[0, 1].plot(percentiles, percentile_values, 'bo-', linewidth=2, markersize=8)
    axes[0, 1].set_title('分位点', fontsize=14)
    axes[0, 1].set_xlabel('パーセント')
    axes[0, 1].set_ylabel('強度 (MPa)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ヒストグラムと密度曲線の重ね合わせ
    axes[1, 0].hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black', label='ヒストグラム')
    
    # カーネル密度推定
    kde = stats.gaussian_kde(data)
    x_kde = np.linspace(min(data), max(data), 200)
    axes[1, 0].plot(x_kde, kde(x_kde), 'r-', linewidth=2, label='カーネル密度推定')
    
    axes[1, 0].set_title('ヒストグラムとカーネル密度推定', fontsize=14)
    axes[1, 0].set_xlabel('強度 (MPa)')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 正規確率プロット
    stats.probplot(data, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('正規確率プロット', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """メイン実行関数"""
    print("=== 可視化の例 ===")
    print("Google Colab用のサンプルスクリプト")
    print("=" * 50)
    
    # 1. サンプルデータの生成
    print("1. サンプルデータの生成中...")
    data = generate_sample_data()
    print(f"   生成されたデータ数: {len(data)}")
    print(f"   データの範囲: {np.min(data):.2f} - {np.max(data):.2f} MPa")
    print(f"   平均: {np.mean(data):.2f} MPa, 標準偏差: {np.std(data):.2f} MPa")
    
    # 2. 基本的な可視化
    print("\n2. 基本的な可視化の作成中...")
    try:
        fig1 = create_basic_visualization(data)
        print("   基本的な可視化完了！'basic_visualization.png'に保存されました。")
    except Exception as e:
        print(f"   基本的な可視化エラー: {e}")
    
    # 3. 分布比較の可視化
    print("\n3. 分布比較の可視化の作成中...")
    try:
        fig2 = create_distribution_comparison(data)
        print("   分布比較の可視化完了！'distribution_comparison.png'に保存されました。")
    except Exception as e:
        print(f"   分布比較の可視化エラー: {e}")
    
    # 4. 統計的サマリーの可視化
    print("\n4. 統計的サマリーの可視化の作成中...")
    try:
        fig3 = create_statistical_summary(data)
        print("   統計的サマリーの可視化完了！'statistical_summary.png'に保存されました。")
    except Exception as e:
        print(f"   統計的サマリーの可視化エラー: {e}")
    
    # 5. 結果の要約
    print("\n5. 結果の要約")
    print("=" * 50)
    
    print("生成された画像ファイル:")
    print("  - basic_visualization.png: 基本的な可視化")
    print("  - distribution_comparison.png: 分布比較")
    print("  - statistical_summary.png: 統計的サマリー")
    
    print("\n可視化の特徴:")
    print("  - ヒストグラムと密度曲線の比較")
    print("  - 複数の分布モデルとのフィッティング比較")
    print("  - 統計的指標の包括的表示")
    print("  - 高解像度（300 DPI）での保存")
    
    print("\n=== 可視化例完了 ===")
    print("このスクリプトはGoogle Colabで完全に実行可能です。")

if __name__ == "__main__":
    main()