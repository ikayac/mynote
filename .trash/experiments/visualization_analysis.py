#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
強度のヒストグラムと様々な分布への当てはめを行った確率密度曲線を重ねて、フィット具合を可視化するスクリプト
"""

import math
import random
import statistics
from collections import defaultdict

def normal_pdf(x, mu, sigma):
    """正規分布の確率密度関数"""
    if sigma <= 0:
        return 0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))

def normal_cdf(x, mu, sigma):
    """正規分布の累積分布関数（近似）"""
    if sigma <= 0:
        return 0
    z = (x - mu) / sigma
    if z < -8:
        return 0
    elif z > 8:
        return 1
    else:
        # 近似式（Hart近似）
        b1 = 0.31938153
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        p = 0.2316419
        c = 0.39894228
        
        if z >= 0:
            t = 1.0 / (1.0 + p * z)
            return 1 - c * math.exp(-z * z / 2) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1)
        else:
            t = 1.0 / (1.0 - p * z)
            return c * math.exp(-z * z / 2) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1)

def generate_mixture_truncated_normal(n_samples=135):
    """270と300付近に頻度が凸となる混合切断正規分布のデータを生成"""
    random.seed(42)
    
    # パラメータ設定
    mu1, sigma1 = 270, 15  # 第1ピーク（270付近）
    mu2, sigma2 = 300, 12  # 第2ピーク（300付近）
    weights = [0.6, 0.4]   # 混合比率
    
    # 切断点
    truncation_point = 235
    
    # 各成分からサンプリング
    n1 = int(n_samples * weights[0])
    n2 = n_samples - n1
    
    # Box-Muller変換で正規分布を生成
    def box_muller(mu, sigma):
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + sigma * z0
    
    # 第1成分（270付近）
    samples1 = []
    while len(samples1) < n1:
        sample = box_muller(mu1, sigma1)
        if sample >= truncation_point:
            samples1.append(sample)
    
    # 第2成分（300付近）
    samples2 = []
    while len(samples2) < n2:
        sample = box_muller(mu2, sigma2)
        if sample >= truncation_point:
            samples2.append(sample)
    
    # 結合
    data = samples1 + samples2
    random.shuffle(data)
    
    return data

def fit_single_distributions(data):
    """単一分布のフィッティング"""
    results = {}
    
    # 基本統計量
    n = len(data)
    mean = statistics.mean(data)
    std = statistics.stdev(data) if n > 1 else 0
    min_val = min(data)
    max_val = max(data)
    
    # 1. 単一正規分布
    results['single_normal'] = {
        'params': {'mu': mean, 'sigma': std},
        'n_params': 2,
        'type': 'single_normal'
    }
    
    # 2. 切断正規分布
    results['truncated_normal'] = {
        'params': {'mu': mean, 'sigma': std},
        'n_params': 2,
        'type': 'truncated_normal'
    }
    
    # 3. ワイブル分布
    shape = 2.0
    loc = min_val
    scale = max(mean - min_val, 1.0)
    results['weibull'] = {
        'params': {'shape': shape, 'loc': loc, 'scale': scale},
        'n_params': 3,
        'type': 'weibull'
    }
    
    # 4. 対数正規分布
    log_data = [math.log(x) for x in data if x > 0]
    if len(log_data) > 0:
        mu_log = statistics.mean(log_data)
        sigma_log = statistics.stdev(log_data) if len(log_data) > 1 else 1.0
    else:
        mu_log = 0
        sigma_log = 1.0
    
    results['lognormal'] = {
        'params': {'shape': sigma_log, 'loc': 0, 'scale': math.exp(mu_log)},
        'n_params': 3,
        'type': 'lognormal'
    }
    
    # 5. ガンマ分布
    if std > 0:
        shape_gamma = (mean - min_val) ** 2 / (std ** 2)
        scale_gamma = (std ** 2) / (mean - min_val)
    else:
        shape_gamma = 1.0
        scale_gamma = 1.0
    
    results['gamma'] = {
        'params': {'shape': shape_gamma, 'loc': min_val, 'scale': scale_gamma},
        'n_params': 3,
        'type': 'gamma'
    }
    
    return results

def fit_mixture_models(data):
    """混合分布モデルのフィッティング"""
    models = {}
    
    # 簡易的なクラスタリングによる初期値推定
    sorted_data = sorted(data)
    n = len(data)
    group_size = n // 2
    
    # 混合正規分布（2成分）
    group1 = sorted_data[:group_size]
    group2 = sorted_data[group_size:]
    
    mu1 = statistics.mean(group1)
    sigma1 = statistics.stdev(group1) if len(group1) > 1 else 1.0
    mu2 = statistics.mean(group2)
    sigma2 = statistics.stdev(group2) if len(group2) > 1 else 1.0
    weight1 = len(group1) / n
    weight2 = len(group2) / n
    
    models['mixture_normal_2'] = {
        'params': {'mus': [mu1, mu2], 'sigmas': [sigma1, sigma2], 'weights': [weight1, weight2]},
        'n_params': 5,
        'type': 'mixture_normal'
    }
    
    # 混合切断正規分布（2成分）
    models['mixture_truncated_normal'] = {
        'params': {'mus': [mu1, mu2], 'sigmas': [sigma1, sigma2], 'weights': [weight1, weight2]},
        'n_params': 5,
        'type': 'mixture_truncated_normal'
    }
    
    return models

def calculate_density_curves(x_range, model_name, params):
    """各モデルの確率密度曲線を計算"""
    
    if model_name == 'single_normal':
        mu, sigma = params['mu'], params['sigma']
        return [normal_pdf(x, mu, sigma) for x in x_range]
    
    elif model_name == 'truncated_normal':
        mu, sigma = params['mu'], params['sigma']
        truncation_point = 235
        
        densities = []
        for x in x_range:
            if x >= truncation_point:
                z = (truncation_point - mu) / sigma
                if sigma > 0 and z < 10:
                    norm_factor = 1 - normal_cdf(truncation_point, mu, sigma)
                    if norm_factor > 1e-10:
                        density = normal_pdf(x, mu, sigma) / norm_factor
                        densities.append(density)
                    else:
                        densities.append(0)
                else:
                    densities.append(0)
            else:
                densities.append(0)
        return densities
    
    elif model_name == 'weibull':
        shape, loc, scale = params['shape'], params['loc'], params['scale']
        densities = []
        for x in x_range:
            if x >= loc and scale > 0 and shape > 0:
                z = (x - loc) / scale
                if z > 0:
                    density = (shape / scale) * (z ** (shape - 1)) * math.exp(-(z ** shape))
                    densities.append(density)
                else:
                    densities.append(0)
            else:
                densities.append(0)
        return densities
    
    elif model_name == 'lognormal':
        shape, loc, scale = params['shape'], params['loc'], params['scale']
        densities = []
        for x in x_range:
            if x > loc and scale > 0 and shape > 0:
                z = (math.log(x - loc) - math.log(scale)) / shape
                density = math.exp(-0.5 * z * z) / ((x - loc) * shape * math.sqrt(2 * math.pi))
                densities.append(density)
            else:
                densities.append(0)
        return densities
    
    elif model_name == 'gamma':
        shape, loc, scale = params['shape'], params['loc'], params['scale']
        densities = []
        for x in x_range:
            if x > loc and scale > 0 and shape > 0:
                z = (x - loc) / scale
                if z > 0:
                    # ガンマ関数の近似（簡易版）
                    density = (z ** (shape - 1)) * math.exp(-z) / scale
                    densities.append(density)
                else:
                    densities.append(0)
            else:
                densities.append(0)
        return densities
    
    elif model_name == 'mixture_normal':
        mus, sigmas, weights = params['mus'], params['sigmas'], params['weights']
        densities = []
        for x in x_range:
            density = 0
            for j in range(len(mus)):
                if sigmas[j] > 0:
                    density += weights[j] * normal_pdf(x, mus[j], sigmas[j])
            densities.append(density)
        return densities
    
    elif model_name == 'mixture_truncated_normal':
        mus, sigmas, weights = params['mus'], params['sigmas'], params['weights']
        truncation_point = 235
        
        densities = []
        for x in x_range:
            if x >= truncation_point:
                density = 0
                for j in range(len(mus)):
                    z = (truncation_point - mus[j]) / sigmas[j]
                    if sigmas[j] > 0 and z < 10:
                        norm_factor = 1 - normal_cdf(truncation_point, mus[j], sigmas[j])
                        if norm_factor > 1e-10:
                            density += weights[j] * normal_pdf(x, mus[j], sigmas[j]) / norm_factor
                densities.append(density)
            else:
                densities.append(0)
        return densities
    
    return [0] * len(x_range)

def create_histogram(data, bins=20):
    """ヒストグラムの作成"""
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / bins
    
    histogram = defaultdict(int)
    bin_centers = []
    
    for i in range(bins):
        center = min_val + (i + 0.5) * bin_width
        bin_centers.append(center)
    
    for value in data:
        bin_index = int((value - min_val) / bin_width)
        if bin_index >= bins:
            bin_index = bins - 1
        histogram[bin_index] += 1
    
    return bin_centers, [histogram[i] for i in range(bins)]

def create_visualization_data(data, models):
    """可視化用のデータを作成"""
    
    # ヒストグラムの作成
    bin_centers, bin_counts = create_histogram(data, 20)
    
    # 密度曲線用のx軸
    x_range = [x for x in range(235, int(max(data)) + 25, 1)]
    
    # 各モデルの密度曲線を計算
    density_curves = {}
    for name, model in models.items():
        densities = calculate_density_curves(x_range, name, model['params'])
        density_curves[name] = densities
    
    return {
        'histogram': {'centers': bin_centers, 'counts': bin_counts},
        'x_range': x_range,
        'density_curves': density_curves,
        'models': models
    }

def generate_visualization_code(data, models):
    """可視化用のPythonコードを生成"""
    
    viz_data = create_visualization_data(data, models)
    
    # 可視化用のPythonコード
    code = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
強度のヒストグラムと様々な分布への当てはめを行った確率密度曲線を重ねて、フィット具合を可視化
"""

import matplotlib.pyplot as plt
import numpy as np

# データ
data = {data}

# ヒストグラムデータ
histogram_centers = {viz_data['histogram']['centers']}
histogram_counts = {viz_data['histogram']['counts']}

# 密度曲線用のx軸
x_range = {viz_data['x_range']}

# 各モデルの密度曲線データ
density_curves = {viz_data['density_curves']}

# 可視化
plt.figure(figsize=(15, 10))

# ヒストグラム（密度に正規化）
plt.subplot(2, 2, 1)
plt.hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black', label='実際のデータ')
plt.title('ヒストグラムと密度曲線の比較（全体）', fontsize=14)
plt.xlabel('強度 (MPa)')
plt.ylabel('密度')

# 各モデルの密度曲線を描画
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
for i, (name, densities) in enumerate(density_curves.items()):
    if densities and any(d > 0 for d in densities):
        plt.plot(x_range, densities, color=colors[i % len(colors)], linewidth=2, label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 270付近の詳細表示
plt.subplot(2, 2, 2)
plt.hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black', label='実際のデータ')
plt.title('270MPa付近の詳細（第1ピーク）', fontsize=14)
plt.xlabel('強度 (MPa)')
plt.ylabel('密度')
plt.xlim(250, 290)

for i, (name, densities) in enumerate(density_curves.items()):
    if densities and any(d > 0 for d in densities):
        plt.plot(x_range, densities, color=colors[i % len(colors)], linewidth=2, label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 300付近の詳細表示
plt.subplot(2, 2, 3)
plt.hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black', label='実際のデータ')
plt.title('300MPa付近の詳細（第2ピーク）', fontsize=14)
plt.xlabel('強度 (MPa)')
plt.ylabel('密度')
plt.xlim(280, 320)

for i, (name, densities) in enumerate(density_curves.items()):
    if densities and any(d > 0 for d in densities):
        plt.plot(x_range, densities, color=colors[i % len(colors)], linewidth=2, label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 下側尾部の詳細表示
plt.subplot(2, 2, 4)
plt.hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black', label='実際のデータ')
plt.title('下側尾部の詳細（235-250MPa）', fontsize=14)
plt.xlabel('強度 (MPa)')
plt.ylabel('密度')
plt.xlim(235, 250)

for i, (name, densities) in enumerate(density_curves.items()):
    if densities and any(d > 0 for d in densities):
        plt.plot(x_range, densities, color=colors[i % len(colors)], linewidth=2, label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('strength_distribution_fit.png', dpi=300, bbox_inches='tight')
plt.show()

print("可視化完了！'strength_distribution_fit.png'に保存されました。")
'''
    
    return code

def main():
    print("強度のヒストグラムと様々な分布への当てはめを行った確率密度曲線を重ねて、フィット具合を可視化するスクリプトを生成します...")
    
    # データ生成
    data = generate_mixture_truncated_normal(135)
    print(f"生成されたデータ数: {len(data)}")
    print(f"データの範囲: {min(data):.2f} - {max(data):.2f}")
    print(f"平均: {statistics.mean(data):.2f}, 標準偏差: {statistics.stdev(data):.2f}")
    
    # 各モデルのフィッティング
    print("\n各モデルのフィッティング中...")
    
    # 単一分布モデル
    single_models = fit_single_distributions(data)
    
    # 混合分布モデル
    mixture_models = fit_mixture_models(data)
    
    # 全モデルを結合
    all_models = {**single_models, **mixture_models}
    
    # 可視化用のコードを生成
    print("\n可視化用のPythonコードを生成中...")
    viz_code = generate_visualization_code(data, all_models)
    
    # コードをファイルに保存
    with open('strength_visualization.py', 'w', encoding='utf-8') as f:
        f.write(viz_code)
    
    print("可視化用のコードを'strength_visualization.py'に保存しました。")
    
    # 可視化データの概要を表示
    viz_data = create_visualization_data(data, all_models)
    
    print(f"\n=== 可視化データの概要 ===")
    print(f"ヒストグラムのビン数: {len(viz_data['histogram']['centers'])}")
    print(f"密度曲線のx軸範囲: {min(viz_data['x_range'])} - {max(viz_data['x_range'])} MPa")
    print(f"密度曲線の点数: {len(viz_data['x_range'])}")
    print(f"モデル数: {len(viz_data['models'])}")
    
    # 各モデルの密度曲線の特徴
    print(f"\n=== 各モデルの密度曲線の特徴 ===")
    for name, densities in viz_data['density_curves'].items():
        if densities:
            max_density = max(densities)
            min_density = min(d for d in densities if d > 0)
            print(f"{name}: 最大密度={max_density:.6f}, 最小密度={min_density:.6f}")
        else:
            print(f"{name}: 密度曲線なし")
    
    print(f"\n可視化を実行するには、以下のコマンドを実行してください:")
    print(f"python3 strength_visualization.py")

if __name__ == "__main__":
    main()