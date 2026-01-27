#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合切断正規分布のデータ生成と様々な分布モデルの当てはめ比較
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def generate_mixture_truncated_normal(n_samples=135):
    """
    270と300付近に頻度が凸となる混合切断正規分布のデータを生成
    """
    np.random.seed(42)
    
    # パラメータ設定
    mu1, sigma1 = 270, 15  # 第1ピーク（270付近）
    mu2, sigma2 = 300, 12  # 第2ピーク（300付近）
    weights = [0.6, 0.4]   # 混合比率
    
    # 切断点
    truncation_point = 235
    
    # 各成分からサンプリング
    n1 = int(n_samples * weights[0])
    n2 = n_samples - n1
    
    # 第1成分（270付近）
    samples1 = np.random.normal(mu1, sigma1, n1 * 3)
    samples1 = samples1[samples1 >= truncation_point][:n1]
    
    # 第2成分（300付近）
    samples2 = np.random.normal(mu2, sigma2, n2 * 3)
    samples2 = samples2[samples2 >= truncation_point][:n2]
    
    # 必要に応じて追加サンプリング
    while len(samples1) < n1:
        temp = np.random.normal(mu1, sigma1, 100)
        temp = temp[temp >= truncation_point]
        samples1 = np.concatenate([samples1, temp])
        if len(samples1) >= n1:
            samples1 = samples1[:n1]
            break
    
    while len(samples2) < n2:
        temp = np.random.normal(mu2, sigma2, 100)
        temp = temp[temp >= truncation_point]
        samples2 = np.concatenate([samples2, temp])
        if len(samples2) >= n2:
            samples2 = samples2[:n2]
            break
    
    # 結合
    data = np.concatenate([samples1, samples2])
    np.random.shuffle(data)
    
    return data

def fit_truncated_normal(data, truncation_point=235):
    """切断正規分布のフィッティング"""
    def neg_log_likelihood(params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        
        # 切断正規分布の対数尤度
        z = (truncation_point - mu) / sigma
        log_likelihood = -0.5 * np.sum(((data - mu) / sigma) ** 2) - len(data) * np.log(sigma)
        log_likelihood -= len(data) * np.log(1 - stats.norm.cdf(z))
        return -log_likelihood
    
    # 初期値
    mu_init = np.mean(data)
    sigma_init = np.std(data)
    
    # 最適化
    result = minimize(neg_log_likelihood, [mu_init, sigma_init], method='L-BFGS-B')
    
    if result.success:
        return result.x[0], result.x[1]
    else:
        return mu_init, sigma_init

def fit_mixture_truncated_normal(data, n_components=2, truncation_point=235):
    """混合切断正規分布のフィッティング"""
    def neg_log_likelihood(params):
        n_params = 3 * n_components  # mu, sigma, weight for each component
        mus = params[:n_components]
        sigmas = params[n_components:2*n_components]
        weights = params[2*n_components:]
        
        # 重みの正規化
        weights = np.exp(weights)
        weights = weights / np.sum(weights)
        
        if np.any(sigmas <= 0):
            return np.inf
        
        log_likelihood = 0
        for i in range(len(data)):
            component_likelihood = 0
            for j in range(n_components):
                z = (truncation_point - mus[j]) / sigmas[j]
                if z < 10:  # 数値的安定性のため
                    norm_factor = 1 - stats.norm.cdf(z)
                    if norm_factor > 1e-10:
                        component_likelihood += weights[j] * stats.norm.pdf(data[i], mus[j], sigmas[j]) / norm_factor
            
            if component_likelihood > 0:
                log_likelihood += np.log(component_likelihood)
        
        return -log_likelihood
    
    # 初期値（GMMで推定）
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data.reshape(-1, 1))
    
    mus_init = gmm.means_.flatten()
    sigmas_init = np.sqrt(gmm.covariances_).flatten()
    weights_init = gmm.weights_
    
    # パラメータの初期化
    params_init = np.concatenate([mus_init, sigmas_init, np.log(weights_init)])
    
    # 最適化
    result = minimize(neg_log_likelihood, params_init, method='L-BFGS-B')
    
    if result.success:
        params = result.x
        mus = params[:n_components]
        sigmas = params[n_components:2*n_components]
        weights = np.exp(params[2*n_components:])
        weights = weights / np.sum(weights)
        return mus, sigmas, weights
    else:
        return mus_init, sigmas_init, weights_init

def fit_mixture_normal(data, n_components=2):
    """混合正規分布のフィッティング"""
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data.reshape(-1, 1))
    return gmm.means_.flatten(), np.sqrt(gmm.covariances_).flatten(), gmm.weights_

def calculate_aic(log_likelihood, n_params):
    """AICの計算"""
    return 2 * n_params - 2 * log_likelihood

def main():
    print("混合切断正規分布のデータ生成とモデル比較を開始します...")
    
    # データ生成
    data = generate_mixture_truncated_normal(135)
    print(f"生成されたデータ数: {len(data)}")
    print(f"データの範囲: {data.min():.2f} - {data.max():.2f}")
    print(f"平均: {np.mean(data):.2f}, 標準偏差: {np.std(data):.2f}")
    
    # 各種分布のフィッティング
    models = {}
    
    # 1. 切断正規分布
    print("\n1. 切断正規分布をフィッティング中...")
    mu_trunc, sigma_trunc = fit_truncated_normal(data)
    models['truncated_normal'] = {
        'params': (mu_trunc, sigma_trunc),
        'n_params': 2,
        'type': 'truncated_normal'
    }
    
    # 2. 混合正規分布（2成分）
    print("2. 混合正規分布（2成分）をフィッティング中...")
    mu_mix, sigma_mix, weights_mix = fit_mixture_normal(data, 2)
    models['mixture_normal_2'] = {
        'params': (mu_mix, sigma_mix, weights_mix),
        'n_params': 5,  # 2*(mu+sigma) + 1(weight)
        'type': 'mixture_normal'
    }
    
    # 3. 混合正規分布（3成分）
    print("3. 混合正規分布（3成分）をフィッティング中...")
    mu_mix3, sigma_mix3, weights_mix3 = fit_mixture_normal(data, 3)
    models['mixture_normal_3'] = {
        'params': (mu_mix3, sigma_mix3, weights_mix3),
        'n_params': 8,  # 3*(mu+sigma) + 2(weights)
        'type': 'mixture_normal'
    }
    
    # 4. 混合切断正規分布（2成分）
    print("4. 混合切断正規分布（2成分）をフィッティング中...")
    mu_mix_trunc, sigma_mix_trunc, weights_mix_trunc = fit_mixture_truncated_normal(data, 2)
    models['mixture_truncated_normal'] = {
        'params': (mu_mix_trunc, sigma_mix_trunc, weights_mix_trunc),
        'n_params': 5,  # 2*(mu+sigma) + 1(weight)
        'type': 'mixture_truncated_normal'
    }
    
    # 5. ワイブル分布
    print("5. ワイブル分布をフィッティング中...")
    shape, loc, scale = stats.weibull_min.fit(data)
    models['weibull'] = {
        'params': (shape, loc, scale),
        'n_params': 3,
        'type': 'weibull'
    }
    
    # 6. 対数正規分布
    print("6. 対数正規分布をフィッティング中...")
    shape_log, loc_log, scale_log = stats.lognorm.fit(data)
    models['lognormal'] = {
        'params': (shape_log, loc_log, scale_log),
        'n_params': 3,
        'type': 'lognormal'
    }
    
    # 7. ガンマ分布
    print("7. ガンマ分布をフィッティング中...")
    shape_gamma, loc_gamma, scale_gamma = stats.gamma.fit(data)
    models['gamma'] = {
        'params': (shape_gamma, loc_gamma, scale_gamma),
        'n_params': 3,
        'type': 'gamma'
    }
    
    # 対数尤度とAICの計算
    print("\n各モデルの対数尤度とAICを計算中...")
    x_range = np.linspace(235, data.max() + 20, 1000)
    
    for name, model in models.items():
        if model['type'] == 'truncated_normal':
            mu, sigma = model['params']
            # 切断正規分布の対数尤度
            z = (235 - mu) / sigma
            log_likelihood = -0.5 * np.sum(((data - mu) / sigma) ** 2) - len(data) * np.log(sigma)
            log_likelihood -= len(data) * np.log(1 - stats.norm.cdf(z))
            
        elif model['type'] == 'mixture_normal':
            mus, sigmas, weights = model['params']
            # 混合正規分布の対数尤度
            log_likelihood = 0
            for i in range(len(data)):
                component_likelihood = 0
                for j in range(len(mus)):
                    component_likelihood += weights[j] * stats.norm.pdf(data[i], mus[j], sigmas[j])
                if component_likelihood > 0:
                    log_likelihood += np.log(component_likelihood)
                    
        elif model['type'] == 'mixture_truncated_normal':
            mus, sigmas, weights = model['params']
            # 混合切断正規分布の対数尤度
            log_likelihood = 0
            for i in range(len(data)):
                component_likelihood = 0
                for j in range(len(mus)):
                    z = (235 - mus[j]) / sigmas[j]
                    if z < 10:
                        norm_factor = 1 - stats.norm.cdf(z)
                        if norm_factor > 1e-10:
                            component_likelihood += weights[j] * stats.norm.pdf(data[i], mus[j], sigmas[j]) / norm_factor
                if component_likelihood > 0:
                    log_likelihood += np.log(component_likelihood)
                    
        elif model['type'] == 'weibull':
            shape, loc, scale = model['params']
            log_likelihood = np.sum(stats.weibull_min.logpdf(data, shape, loc, scale))
            
        elif model['type'] == 'lognormal':
            shape, loc, scale = model['params']
            log_likelihood = np.sum(stats.lognorm.logpdf(data, shape, loc, scale))
            
        elif model['type'] == 'gamma':
            shape, loc, scale = model['params']
            log_likelihood = np.sum(stats.gamma.logpdf(data, shape, loc, scale))
        
        model['log_likelihood'] = log_likelihood
        model['aic'] = calculate_aic(log_likelihood, model['n_params'])
        
        print(f"{name}: AIC = {model['aic']:.2f}, Log-Likelihood = {log_likelihood:.2f}")
    
    # AICの昇順でソート
    sorted_models = sorted(models.items(), key=lambda x: x[1]['aic'])
    
    print(f"\n=== AICによるモデルランキング ===")
    for i, (name, model) in enumerate(sorted_models):
        print(f"{i+1}. {name}: AIC = {model['aic']:.2f}")
    
    # 可視化
    print("\n可視化を生成中...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('混合切断正規分布データと各種分布モデルの比較', fontsize=16)
    
    # 1. ヒストグラムと密度推定
    ax1 = axes[0, 0]
    ax1.hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_title('データのヒストグラムと密度推定')
    ax1.set_xlabel('強度 (MPa)')
    ax1.set_ylabel('密度')
    
    # 各モデルの密度曲線を描画
    x_range = np.linspace(235, data.max() + 20, 1000)
    
    for name, model in sorted_models[:3]:  # 上位3モデルのみ表示
        if model['type'] == 'truncated_normal':
            mu, sigma = model['params']
            z = (235 - mu) / sigma
            norm_factor = 1 - stats.norm.cdf(z)
            density = stats.norm.pdf(x_range, mu, sigma) / norm_factor
            ax1.plot(x_range, density, label=f'{name} (AIC: {model["aic"]:.1f})', linewidth=2)
            
        elif model['type'] == 'mixture_normal':
            mus, sigmas, weights = model['params']
            density = np.zeros_like(x_range)
            for j in range(len(mus)):
                density += weights[j] * stats.norm.pdf(x_range, mus[j], sigmas[j])
            ax1.plot(x_range, density, label=f'{name} (AIC: {model["aic"]:.1f})', linewidth=2)
            
        elif model['type'] == 'mixture_truncated_normal':
            mus, sigmas, weights = model['params']
            density = np.zeros_like(x_range)
            for j in range(len(mus)):
                z = (235 - mus[j]) / sigmas[j]
                norm_factor = 1 - stats.norm.cdf(z)
                density += weights[j] * stats.norm.pdf(x_range, mus[j], sigmas[j]) / norm_factor
            ax1.plot(x_range, density, label=f'{name} (AIC: {model["aic"]:.1f})', linewidth=2)
            
        elif model['type'] == 'weibull':
            shape, loc, scale = model['params']
            density = stats.weibull_min.pdf(x_range, shape, loc, scale)
            ax1.plot(x_range, density, label=f'{name} (AIC: {model["aic"]:.1f})', linewidth=2)
            
        elif model['type'] == 'lognormal':
            shape, loc, scale = model['params']
            density = stats.lognorm.pdf(x_range, shape, loc, scale)
            ax1.plot(x_range, density, label=f'{name} (AIC: {model["aic"]:.1f})', linewidth=2)
            
        elif model['type'] == 'gamma':
            shape, loc, scale = model['params']
            density = stats.gamma.pdf(x_range, shape, loc, scale)
            ax1.plot(x_range, density, label=f'{name} (AIC: {model["aic"]:.1f})', linewidth=2)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Qプロット（最良モデル）
    ax2 = axes[0, 1]
    best_model_name = sorted_models[0][0]
    best_model = sorted_models[0][1]
    
    if best_model['type'] == 'mixture_truncated_normal':
        # 混合切断正規分布のQ-Qプロット
        mus, sigmas, weights = best_model['params']
        theoretical_quantiles = []
        empirical_quantiles = np.sort(data)
        
        for i, p in enumerate(np.linspace(0.01, 0.99, len(data))):
            # 混合切断正規分布の分位点を計算
            quantile = 0
            for j in range(len(mus)):
                z = (235 - mus[j]) / sigmas[j]
                norm_factor = 1 - stats.norm.cdf(z)
                if norm_factor > 1e-10:
                    # 切断正規分布の分位点
                    p_adj = p * norm_factor + stats.norm.cdf(z)
                    quantile += weights[j] * stats.norm.ppf(p_adj, mus[j], sigmas[j])
            theoretical_quantiles.append(quantile)
        
        theoretical_quantiles = np.array(theoretical_quantiles)
        ax2.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.7)
        ax2.plot([theoretical_quantiles.min(), theoretical_quantiles.max()], 
                [theoretical_quantiles.min(), theoretical_quantiles.max()], 'r--', linewidth=2)
        ax2.set_title(f'Q-Qプロット: {best_model_name}')
        ax2.set_xlabel('理論的分位点')
        ax2.set_ylabel('経験的分位点')
    
    ax2.grid(True, alpha=0.3)
    
    # 3. AIC比較
    ax3 = axes[1, 0]
    names = [name for name, _ in sorted_models]
    aics = [model['aic'] for _, model in sorted_models]
    
    bars = ax3.bar(range(len(names)), aics, color='skyblue', alpha=0.7)
    ax3.set_title('各モデルのAIC比較')
    ax3.set_xlabel('モデル')
    ax3.set_ylabel('AIC')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right')
    
    # 最良モデルを強調
    bars[0].set_color('red')
    bars[0].set_alpha(0.8)
    
    ax3.grid(True, alpha=0.3)
    
    # 4. 残差分析
    ax4 = axes[1, 1]
    if best_model['type'] == 'mixture_truncated_normal':
        mus, sigmas, weights = best_model['params']
        # 混合切断正規分布の累積分布関数
        cdf_values = []
        for x in np.sort(data):
            cdf = 0
            for j in range(len(mus)):
                z = (235 - mus[j]) / sigmas[j]
                norm_factor = 1 - stats.norm.cdf(z)
                if norm_factor > 1e-10:
                    cdf += weights[j] * (stats.norm.cdf(x, mus[j], sigmas[j]) - stats.norm.cdf(235, mus[j], sigmas[j])) / norm_factor
            cdf_values.append(cdf)
        
        cdf_values = np.array(cdf_values)
        empirical_cdf = np.linspace(1/len(data), 1, len(data))
        
        ax4.scatter(cdf_values, empirical_cdf, alpha=0.7)
        ax4.plot([0, 1], [0, 1], 'r--', linewidth=2)
        ax4.set_title(f'P-Pプロット: {best_model_name}')
        ax4.set_xlabel('理論的累積確率')
        ax4.set_ylabel('経験的累積確率')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n可視化結果を 'distribution_comparison.png' に保存しました。")
    print(f"最良モデル: {best_model_name} (AIC: {best_model['aic']:.2f})")
    
    # 詳細な結果をCSVに保存
    import pandas as pd
    results_df = pd.DataFrame([
        {
            'Model': name,
            'AIC': model['aic'],
            'Log_Likelihood': model['log_likelihood'],
            'Parameters': str(model['n_params'])
        }
        for name, model in sorted_models
    ])
    
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("詳細結果を 'model_comparison_results.csv' に保存しました。")

if __name__ == "__main__":
    main()