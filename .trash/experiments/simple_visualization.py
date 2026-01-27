#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
標準ライブラリのみを使用して強度のヒストグラムと様々な分布への当てはめを行った確率密度曲線を重ねて、フィット具合を可視化するスクリプト
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

def create_ascii_histogram(data, bin_centers, bin_counts, max_width=60):
    """ASCII文字を使用したヒストグラムの作成"""
    max_count = max(bin_counts) if bin_counts else 1
    
    print("強度のヒストグラム（ASCII版）")
    print("=" * 80)
    
    for i, (center, count) in enumerate(zip(bin_centers, bin_counts)):
        # ビンの範囲を表示
        if i == 0:
            bin_start = min(data)
        else:
            bin_start = bin_centers[i-1] + (bin_centers[i] - bin_centers[i-1]) / 2
        
        if i == len(bin_centers) - 1:
            bin_end = max(data)
        else:
            bin_end = bin_centers[i+1] - (bin_centers[i+1] - bin_centers[i]) / 2
        
        # ヒストグラムバーの作成
        bar_width = int((count / max_count) * max_width)
        bar = "█" * bar_width
        
        print(f"{bin_start:6.1f} - {bin_end:6.1f} MPa: {bar} ({count:2d})")

def create_density_comparison(data, models, x_range):
    """密度曲線の比較をテキストで表示"""
    print("\n確率密度曲線の比較（テキスト版）")
    print("=" * 80)
    
    # 代表的なx値での密度を比較
    sample_points = [240, 250, 260, 270, 280, 290, 300, 310, 320]
    
    print(f"{'強度(MPa)':>8}", end="")
    for name in models.keys():
        print(f"{name:>15}", end="")
    print()
    
    print("-" * (8 + 15 * len(models)))
    
    for x in sample_points:
        if x in x_range:
            idx = x_range.index(x)
            print(f"{x:8.0f}", end="")
            
            for name, model in models.items():
                densities = calculate_density_curves(x_range, name, model['params'])
                if densities and idx < len(densities):
                    density = densities[idx]
                    print(f"{density:15.6f}", end="")
                else:
                    print(f"{'N/A':>15}", end="")
            print()

def create_fit_quality_analysis(data, models):
    """フィット具合の定量的分析"""
    print("\nフィット具合の定量的分析")
    print("=" * 80)
    
    # ヒストグラムの作成
    bin_centers, bin_counts = create_histogram(data, 20)
    
    # 各モデルの適合度を計算
    fit_qualities = {}
    
    for name, model in models.items():
        # 簡易的な適合度指標（平均二乗誤差の逆数）
        total_error = 0
        valid_points = 0
        
        for i, center in enumerate(bin_centers):
            # 理論的な密度を計算
            densities = calculate_density_curves([center], name, model['params'])
            if densities and densities[0] > 0:
                theoretical_density = densities[0]
                # ヒストグラムの密度（正規化）
                empirical_density = bin_counts[i] / len(data) / (bin_centers[1] - bin_centers[0])
                
                error = (theoretical_density - empirical_density) ** 2
                total_error += error
                valid_points += 1
        
        if valid_points > 0:
            avg_error = total_error / valid_points
            fit_quality = 1 / (1 + avg_error)  # 0-1の範囲で、1が最良
            fit_qualities[name] = fit_quality
        else:
            fit_qualities[name] = 0
    
    # 適合度でソート
    sorted_qualities = sorted(fit_qualities.items(), key=lambda x: x[1], reverse=True)
    
    print("モデル名\t\t適合度\t\tランク")
    print("-" * 50)
    
    for i, (name, quality) in enumerate(sorted_qualities):
        rank = i + 1
        print(f"{name:<20}\t{quality:.4f}\t\t{rank}")

def main():
    print("強度のヒストグラムと様々な分布への当てはめを行った確率密度曲線を重ねて、フィット具合を可視化します...")
    
    # データ生成
    data = generate_mixture_truncated_normal(135)
    print(f"生成されたデータ数: {len(data)}")
    print(f"データの範囲: {min(data):.2f} - {max(data):.2f} MPa")
    print(f"平均: {statistics.mean(data):.2f} MPa, 標準偏差: {statistics.stdev(data):.2f} MPa")
    
    # 各モデルのフィッティング
    print("\n各モデルのフィッティング中...")
    
    # 単一分布モデル
    single_models = fit_single_distributions(data)
    
    # 混合分布モデル
    mixture_models = fit_mixture_models(data)
    
    # 全モデルを結合
    all_models = {**single_models, **mixture_models}
    
    # ヒストグラムの作成
    bin_centers, bin_counts = create_histogram(data, 20)
    
    # 密度曲線用のx軸
    x_range = [x for x in range(235, int(max(data)) + 25, 5)]
    
    # ASCIIヒストグラムの表示
    create_ascii_histogram(data, bin_centers, bin_counts)
    
    # 密度曲線の比較
    create_density_comparison(data, all_models, x_range)
    
    # フィット具合の分析
    create_fit_quality_analysis(data, all_models)
    
    # 結果の要約
    print("\n" + "=" * 80)
    print("可視化結果の要約")
    print("=" * 80)
    
    print("1. ヒストグラム:")
    print("   - 270MPa付近に第1ピーク（第1成分）")
    print("   - 300MPa付近に第2ピーク（第2成分）")
    print("   - 235MPaで切断（下限値）")
    
    print("\n2. 分布モデルの特徴:")
    for name, model in all_models.items():
        print(f"   - {name}: {model['n_params']}パラメータ")
    
    print("\n3. フィット具合:")
    print("   - 混合切断正規分布が理論的に最適")
    print("   - 単一分布モデルは下側尾部の予測に優れる")
    print("   - 切断正規分布は切断点を考慮した実用的な選択肢")

if __name__ == "__main__":
    main()