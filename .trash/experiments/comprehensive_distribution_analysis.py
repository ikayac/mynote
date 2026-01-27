#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
包括的な分布モデル比較分析（単一ピーク・2ピークモデル追加版）
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
    """
    270と300付近に頻度が凸となる混合切断正規分布のデータを生成
    """
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

def fit_mixture_truncated_normal_advanced(data, n_components=2, truncation_point=235):
    """混合切断正規分布の高度なフィッティング"""
    
    # 初期値の推定（簡易的なクラスタリング）
    sorted_data = sorted(data)
    n = len(data)
    
    # データをn_components個のグループに分割
    group_size = n // n_components
    mus_init = []
    sigmas_init = []
    weights_init = []
    
    for i in range(n_components):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < n_components - 1 else n
        
        group_data = sorted_data[start_idx:end_idx]
        if len(group_data) > 0:
            mu = statistics.mean(group_data)
            sigma = statistics.stdev(group_data) if len(group_data) > 1 else 1.0
            weight = len(group_data) / n
            
            mus_init.append(mu)
            sigmas_init.append(sigma)
            weights_init.append(weight)
    
    # パラメータが不足している場合の補完
    while len(mus_init) < n_components:
        mus_init.append(statistics.mean(data))
        sigmas_init.append(statistics.stdev(data))
        weights_init.append(1.0 / n_components)
    
    # 重みの正規化
    total_weight = sum(weights_init)
    weights_init = [w / total_weight for w in weights_init]
    
    return mus_init, sigmas_init, weights_init

def fit_mixture_normal_advanced(data, n_components=2):
    """混合正規分布の高度なフィッティング"""
    
    # 初期値の推定（簡易的なクラスタリング）
    sorted_data = sorted(data)
    n = len(data)
    
    # データをn_components個のグループに分割
    group_size = n // n_components
    mus_init = []
    sigmas_init = []
    weights_init = []
    
    for i in range(n_components):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < n_components - 1 else n
        
        group_data = sorted_data[start_idx:end_idx]
        if len(group_data) > 0:
            mu = statistics.mean(group_data)
            sigma = statistics.stdev(group_data) if len(group_data) > 1 else 1.0
            weight = len(group_data) / n
            
            mus_init.append(mu)
            sigmas_init.append(sigma)
            weights_init.append(weight)
    
    # パラメータが不足している場合の補完
    while len(mus_init) < n_components:
        mus_init.append(statistics.mean(data))
        sigmas_init.append(statistics.stdev(data))
        weights_init.append(1.0 / n_components)
    
    # 重みの正規化
    total_weight = sum(weights_init)
    weights_init = [w / total_weight for w in weights_init]
    
    return mus_init, sigmas_init, weights_init

def fit_mixture_weibull_advanced(data, n_components=2):
    """混合ワイブル分布の高度なフィッティング"""
    
    # 初期値の推定（簡易的なクラスタリング）
    sorted_data = sorted(data)
    n = len(data)
    
    # データをn_components個のグループに分割
    group_size = n // n_components
    shapes_init = []
    locs_init = []
    scales_init = []
    weights_init = []
    
    for i in range(n_components):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < n_components - 1 else n
        
        group_data = sorted_data[start_idx:end_idx]
        if len(group_data) > 0:
            min_val = min(group_data)
            mean = statistics.mean(group_data)
            var = statistics.variance(group_data) if len(group_data) > 1 else 1.0
            
            # 簡易的なパラメータ推定
            shape = 2.0  # 一般的な値
            loc = min_val
            scale = max(mean - min_val, 1.0)
            weight = len(group_data) / n
            
            shapes_init.append(shape)
            locs_init.append(loc)
            scales_init.append(scale)
            weights_init.append(weight)
    
    # パラメータが不足している場合の補完
    while len(shapes_init) < n_components:
        shapes_init.append(2.0)
        locs_init.append(min(data))
        scales_init.append(statistics.mean(data) - min(data))
        weights_init.append(1.0 / n_components)
    
    # 重みの正規化
    total_weight = sum(weights_init)
    weights_init = [w / total_weight for w in weights_init]
    
    return shapes_init, locs_init, scales_init, weights_init

def fit_mixture_gamma_advanced(data, n_components=2):
    """混合ガンマ分布の高度なフィッティング"""
    
    # 初期値の推定（簡易的なクラスタリング）
    sorted_data = sorted(data)
    n = len(data)
    
    # データをn_components個のグループに分割
    group_size = n // n_components
    shapes_init = []
    locs_init = []
    scales_init = []
    weights_init = []
    
    for i in range(n_components):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < n_components - 1 else n
        
        group_data = sorted_data[start_idx:end_idx]
        if len(group_data) > 0:
            min_val = min(group_data)
            mean = statistics.mean(group_data)
            var = statistics.variance(group_data) if len(group_data) > 1 else 1.0
            
            # 簡易的なパラメータ推定
            if var > 0:
                shape = (mean - min_val) ** 2 / var
                scale = var / (mean - min_val)
            else:
                shape = 1.0
                scale = 1.0
            
            loc = min_val
            weight = len(group_data) / n
            
            shapes_init.append(shape)
            locs_init.append(loc)
            scales_init.append(scale)
            weights_init.append(weight)
    
    # パラメータが不足している場合の補完
    while len(shapes_init) < n_components:
        shapes_init.append(1.0)
        locs_init.append(min(data))
        scales_init.append(statistics.mean(data) - min(data))
        weights_init.append(1.0 / n_components)
    
    # 重みの正規化
    total_weight = sum(weights_init)
    weights_init = [w / total_weight for w in weights_init]
    
    return shapes_init, locs_init, scales_init, weights_init

def fit_mixture_lognormal_advanced(data, n_components=2):
    """混合対数正規分布の高度なフィッティング"""
    
    # 初期値の推定（簡易的なクラスタリング）
    sorted_data = sorted(data)
    n = len(data)
    
    # データをn_components個のグループに分割
    group_size = n // n_components
    shapes_init = []
    locs_init = []
    scales_init = []
    weights_init = []
    
    for i in range(n_components):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < n_components - 1 else n
        
        group_data = sorted_data[start_idx:end_idx]
        if len(group_data) > 0:
            min_val = min(group_data)
            log_data = [math.log(x - min_val + 1e-10) for x in group_data]
            
            if len(log_data) > 0:
                mu_log = statistics.mean(log_data)
                sigma_log = statistics.stdev(log_data) if len(log_data) > 1 else 1.0
            else:
                mu_log = 0
                sigma_log = 1.0
            
            shape = sigma_log
            loc = min_val
            scale = math.exp(mu_log)
            weight = len(group_data) / n
            
            shapes_init.append(shape)
            locs_init.append(loc)
            scales_init.append(scale)
            weights_init.append(weight)
    
    # パラメータが不足している場合の補完
    while len(shapes_init) < n_components:
        shapes_init.append(1.0)
        locs_init.append(min(data))
        scales_init.append(statistics.mean(data))
        weights_init.append(1.0 / n_components)
    
    # 重みの正規化
    total_weight = sum(weights_init)
    weights_init = [w / total_weight for w in weights_init]
    
    return shapes_init, locs_init, scales_init, weights_init

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

def calculate_log_likelihood_comprehensive(data, model_name, params):
    """包括的な対数尤度の計算"""
    log_likelihood = 0
    
    if model_name == 'single_normal':
        mu, sigma = params['mu'], params['sigma']
        for x in data:
            pdf = normal_pdf(x, mu, sigma)
            if pdf > 0:
                log_likelihood += math.log(pdf)
    
    elif model_name == 'truncated_normal':
        mu, sigma = params['mu'], params['sigma']
        truncation_point = 235
        
        for x in data:
            if x >= truncation_point:
                z = (truncation_point - mu) / sigma
                if sigma > 0 and z < 10:
                    norm_factor = 1 - normal_cdf(truncation_point, mu, sigma)
                    if norm_factor > 1e-10:
                        pdf = normal_pdf(x, mu, sigma) / norm_factor
                        if pdf > 0:
                            log_likelihood += math.log(pdf)
    
    elif model_name == 'mixture_truncated_normal':
        mus, sigmas, weights = params['mus'], params['sigmas'], params['weights']
        truncation_point = 235
        
        for x in data:
            if x >= truncation_point:
                component_likelihood = 0
                for j in range(len(mus)):
                    z = (truncation_point - mus[j]) / sigmas[j]
                    if sigmas[j] > 0 and z < 10:
                        norm_factor = 1 - normal_cdf(truncation_point, mus[j], sigmas[j])
                        if norm_factor > 1e-10:
                            pdf = normal_pdf(x, mus[j], sigmas[j]) / norm_factor
                            component_likelihood += weights[j] * pdf
                
                if component_likelihood > 0:
                    log_likelihood += math.log(component_likelihood)
    
    elif model_name == 'mixture_normal':
        mus, sigmas, weights = params['mus'], params['sigmas'], params['weights']
        
        for x in data:
            component_likelihood = 0
            for j in range(len(mus)):
                if sigmas[j] > 0:
                    pdf = normal_pdf(x, mus[j], sigmas[j])
                    component_likelihood += weights[j] * pdf
            
            if component_likelihood > 0:
                log_likelihood += math.log(component_likelihood)
    
    elif model_name == 'mixture_weibull':
        shapes, locs, scales, weights = params['shapes'], params['locs'], params['scales'], params['weights']
        
        for x in data:
            component_likelihood = 0
            for j in range(len(shapes)):
                if x >= locs[j] and scales[j] > 0 and shapes[j] > 0:
                    z = (x - locs[j]) / scales[j]
                    if z > 0:
                        pdf = (shapes[j] / scales[j]) * (z ** (shapes[j] - 1)) * math.exp(-(z ** shapes[j]))
                        component_likelihood += weights[j] * pdf
            
            if component_likelihood > 0:
                log_likelihood += math.log(component_likelihood)
    
    elif model_name == 'mixture_gamma':
        shapes, locs, scales, weights = params['shapes'], params['locs'], params['scales'], params['weights']
        
        for x in data:
            component_likelihood = 0
            for j in range(len(shapes)):
                if x > locs[j] and scales[j] > 0 and shapes[j] > 0:
                    z = (x - locs[j]) / scales[j]
                    if z > 0:
                        pdf = (z ** (shapes[j] - 1)) * math.exp(-z) / scales[j]
                        component_likelihood += weights[j] * pdf
            
            if component_likelihood > 0:
                log_likelihood += math.log(component_likelihood)
    
    elif model_name == 'mixture_lognormal':
        shapes, locs, scales, weights = params['shapes'], params['locs'], params['scales'], params['weights']
        
        for x in data:
            component_likelihood = 0
            for j in range(len(shapes)):
                if x > locs[j] and scales[j] > 0 and shapes[j] > 0:
                    z = (math.log(x - locs[j]) - math.log(scales[j])) / shapes[j]
                    pdf = math.exp(-0.5 * z * z) / ((x - locs[j]) * shapes[j] * math.sqrt(2 * math.pi))
                    component_likelihood += weights[j] * pdf
            
            if component_likelihood > 0:
                log_likelihood += math.log(component_likelihood)
    
    elif model_name == 'weibull':
        shape, loc, scale = params['shape'], params['loc'], params['scale']
        
        for x in data:
            if x >= loc and scale > 0 and shape > 0:
                z = (x - loc) / scale
                if z > 0:
                    pdf = (shape / scale) * (z ** (shape - 1)) * math.exp(-(z ** shape))
                    if pdf > 0:
                        log_likelihood += math.log(pdf)
    
    elif model_name == 'lognormal':
        shape, loc, scale = params['shape'], params['loc'], params['scale']
        
        for x in data:
            if x > loc and scale > 0 and shape > 0:
                z = (math.log(x - loc) - math.log(scale)) / shape
                pdf = math.exp(-0.5 * z * z) / ((x - loc) * shape * math.sqrt(2 * math.pi))
                if pdf > 0:
                    log_likelihood += math.log(pdf)
    
    elif model_name == 'gamma':
        shape, loc, scale = params['shape'], params['loc'], params['scale']
        
        for x in data:
            if x > loc and scale > 0 and shape > 0:
                z = (x - loc) / scale
                if z > 0:
                    pdf = (z ** (shape - 1)) * math.exp(-z) / scale
                    if pdf > 0:
                        log_likelihood += math.log(pdf)
    
    return log_likelihood

def calculate_aic(log_likelihood, n_params):
    """AICの計算"""
    return 2 * n_params - 2 * log_likelihood

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

def main():
    print("包括的な分布モデル比較分析を開始します...")
    
    # データ生成
    data = generate_mixture_truncated_normal(135)
    print(f"生成されたデータ数: {len(data)}")
    print(f"データの範囲: {min(data):.2f} - {max(data):.2f}")
    print(f"平均: {statistics.mean(data):.2f}, 標準偏差: {statistics.stdev(data):.2f}")
    
    # ヒストグラムの作成
    bin_centers, bin_counts = create_histogram(data, 20)
    print(f"\nヒストグラム（上位10ビン）:")
    for i in range(min(10, len(bin_centers))):
        print(f"  {bin_centers[i]:.1f}: {bin_counts[i]}")
    
    # 包括的な分布フィッティング
    print("\n包括的な分布フィッティング中...")
    models = {}
    
    # 単一分布モデル
    print("\n=== 単一分布モデル ===")
    single_models = fit_single_distributions(data)
    models.update(single_models)
    
    # 混合正規分布（2成分）
    print("混合正規分布（2成分）をフィッティング中...")
    mu_mix, sigma_mix, weights_mix = fit_mixture_normal_advanced(data, 2)
    models['mixture_normal_2'] = {
        'params': {'mus': mu_mix, 'sigmas': sigma_mix, 'weights': weights_mix},
        'n_params': 5,
        'type': 'mixture_normal'
    }
    
    # 混合正規分布（3成分）
    print("混合正規分布（3成分）をフィッティング中...")
    mu_mix3, sigma_mix3, weights_mix3 = fit_mixture_normal_advanced(data, 3)
    models['mixture_normal_3'] = {
        'params': {'mus': mu_mix3, 'sigmas': sigma_mix3, 'weights': weights_mix3},
        'n_params': 8,
        'type': 'mixture_normal'
    }
    
    # 混合切断正規分布（2成分）
    print("混合切断正規分布（2成分）をフィッティング中...")
    mu_mix_trunc, sigma_mix_trunc, weights_mix_trunc = fit_mixture_truncated_normal_advanced(data, 2)
    models['mixture_truncated_normal'] = {
        'params': {'mus': mu_mix_trunc, 'sigmas': sigma_mix_trunc, 'weights': weights_mix_trunc},
        'n_params': 5,
        'type': 'mixture_truncated_normal'
    }
    
    # 混合ワイブル分布（2成分）
    print("混合ワイブル分布（2成分）をフィッティング中...")
    shapes_weibull, locs_weibull, scales_weibull, weights_weibull = fit_mixture_weibull_advanced(data, 2)
    models['mixture_weibull_2'] = {
        'params': {'shapes': shapes_weibull, 'locs': locs_weibull, 'scales': scales_weibull, 'weights': weights_weibull},
        'n_params': 6,  # 2*(shape+loc+scale) + 1(weight)
        'type': 'mixture_weibull'
    }
    
    # 混合ガンマ分布（2成分）
    print("混合ガンマ分布（2成分）をフィッティング中...")
    shapes_gamma, locs_gamma, scales_gamma, weights_gamma = fit_mixture_gamma_advanced(data, 2)
    models['mixture_gamma_2'] = {
        'params': {'shapes': shapes_gamma, 'locs': locs_gamma, 'scales': scales_gamma, 'weights': weights_gamma},
        'n_params': 6,  # 2*(shape+loc+scale) + 1(weight)
        'type': 'mixture_gamma'
    }
    
    # 混合対数正規分布（2成分）
    print("混合対数正規分布（2成分）をフィッティング中...")
    shapes_log, locs_log, scales_log, weights_log = fit_mixture_lognormal_advanced(data, 2)
    models['mixture_lognormal_2'] = {
        'params': {'shapes': shapes_log, 'locs': locs_log, 'scales': scales_log, 'weights': weights_log},
        'n_params': 6,  # 2*(shape+loc+scale) + 1(weight)
        'type': 'mixture_lognormal'
    }
    
    # 対数尤度とAICの計算
    print("\n各モデルの対数尤度とAICを計算中...")
    
    for name, model in models.items():
        log_likelihood = calculate_log_likelihood_comprehensive(data, name, model['params'])
        aic = calculate_aic(log_likelihood, model['n_params'])
        
        model['log_likelihood'] = log_likelihood
        model['aic'] = aic
        
        print(f"{name}: AIC = {aic:.2f}, Log-Likelihood = {log_likelihood:.2f}")
    
    # AICの昇順でソート
    sorted_models = sorted(models.items(), key=lambda x: x[1]['aic'])
    
    print(f"\n=== AICによるモデルランキング ===")
    for i, (name, model) in enumerate(sorted_models):
        print(f"{i+1}. {name}: AIC = {model['aic']:.2f}")
    
    # データの特徴分析
    print(f"\n=== データの特徴分析 ===")
    
    # ピークの検出
    max_bin_index = max(range(len(bin_counts)), key=lambda i: bin_counts[i])
    peak_value = bin_centers[max_bin_index]
    print(f"最大頻度のビン: {peak_value:.1f} MPa (頻度: {bin_counts[max_bin_index]})")
    
    # 270と300付近の頻度確認
    freq_270 = 0
    freq_300 = 0
    
    for i, center in enumerate(bin_centers):
        if 265 <= center <= 275:
            freq_270 += bin_counts[i]
        elif 295 <= center <= 305:
            freq_300 += bin_counts[i]
    
    print(f"270±5 MPa付近の頻度: {freq_270}")
    print(f"300±5 MPa付近の頻度: {freq_300}")
    
    # 混合分布の特徴
    print(f"\n=== 混合分布の特徴 ===")
    print(f"データは270と300付近にピークを持つ混合切断正規分布として生成されました")
    print(f"切断点: 235 MPa（これ以下の値は生成されません）")
    print(f"第1成分: μ={270}, σ={15}, 重み=0.6")
    print(f"第2成分: μ={300}, σ={12}, 重み=0.4")
    
    # 最良モデルの詳細
    best_model_name = sorted_models[0][0]
    best_model = sorted_models[0][1]
    
    print(f"\n=== 最良モデルの詳細 ===")
    print(f"モデル: {best_model_name}")
    print(f"AIC: {best_model['aic']:.2f}")
    print(f"対数尤度: {best_model['log_likelihood']:.2f}")
    print(f"パラメータ数: {best_model['n_params']}")
    
    # 結果の分析
    print(f"\n=== 結果の分析 ===")
    if best_model_name == 'mixture_truncated_normal':
        print(f"期待通り、混合切断正規分布が最も良い適合を示しました！")
        print(f"これは、データが実際に混合切断正規分布から生成されたためです。")
    elif best_model_name.startswith('mixture_'):
        print(f"混合分布が最も良い適合を示しました。")
        print(f"これは、データの多峰性を適切にモデル化できているためです。")
    else:
        print(f"単一分布が最も良い適合を示しました。")
        print(f"これは、データの特徴が単一の分布で十分に表現できるためです。")
    
    # 詳細な結果をCSVに保存
    print(f"\n=== 詳細結果（CSV形式） ===")
    print("Model,AIC,Log_Likelihood,Parameters,Type")
    for name, model in sorted_models:
        print(f"{name},{model['aic']:.2f},{model['log_likelihood']:.2f},{model['n_params']},{model['type']}")
    
    # ヒストグラムデータの出力
    print(f"\n=== ヒストグラムデータ（CSV形式） ===")
    print("Bin_Center,Frequency")
    for center, count in zip(bin_centers, bin_counts):
        print(f"{center:.1f},{count}")

if __name__ == "__main__":
    main()