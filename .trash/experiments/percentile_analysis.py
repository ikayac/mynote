#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
各分布モデルの5%分位点（下側下限値）を計算するスクリプト
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

def normal_ppf(p, mu, sigma):
    """正規分布の分位点関数（近似）"""
    if p <= 0:
        return mu - 4 * sigma
    elif p >= 1:
        return mu + 4 * sigma
    
    # 標準正規分布の分位点を近似
    if p < 0.5:
        # 0.5未満の場合
        t = math.sqrt(-2 * math.log(p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        
        x = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
    else:
        # 0.5以上の場合
        t = math.sqrt(-2 * math.log(1 - p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        
        x = -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))
    
    return mu + sigma * x

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

def calculate_5th_percentile(model_name, params):
    """各モデルの5%分位点を計算"""
    
    if model_name == 'single_normal':
        mu, sigma = params['mu'], params['sigma']
        return normal_ppf(0.05, mu, sigma)
    
    elif model_name == 'truncated_normal':
        mu, sigma = params['mu'], params['sigma']
        truncation_point = 235
        
        # 切断正規分布の5%分位点
        # 切断点以下の確率を考慮して調整
        z = (truncation_point - mu) / sigma
        if z < 10:
            norm_factor = 1 - normal_cdf(truncation_point, mu, sigma)
            if norm_factor > 1e-10:
                # 切断後の5%分位点
                p_adjusted = 0.05 * norm_factor
                return normal_ppf(p_adjusted, mu, sigma)
        
        return truncation_point
    
    elif model_name == 'weibull':
        shape, loc, scale = params['shape'], params['loc'], params['scale']
        if shape > 0 and scale > 0:
            # ワイブル分布の5%分位点
            return loc + scale * ((-math.log(0.95)) ** (1 / shape))
        return params['loc']
    
    elif model_name == 'lognormal':
        shape, loc, scale = params['shape'], params['loc'], params['scale']
        if shape > 0 and scale > 0:
            # 対数正規分布の5%分位点
            return loc + scale * math.exp(shape * normal_ppf(0.05, 0, 1))
        return params['loc']
    
    elif model_name == 'gamma':
        shape, loc, scale = params['shape'], params['loc'], params['scale']
        if shape > 0 and scale > 0:
            # ガンマ分布の5%分位点（簡易近似）
            # 形状パラメータが大きい場合、正規分布で近似
            if shape > 10:
                mean_gamma = loc + shape * scale
                std_gamma = math.sqrt(shape) * scale
                return normal_ppf(0.05, mean_gamma, std_gamma)
            else:
                # 簡易的な近似
                return loc + scale * (shape - 1.645 * math.sqrt(shape))
        return params['loc']
    
    elif model_name == 'mixture_normal':
        mus, sigmas, weights = params['mus'], params['sigmas'], params['weights']
        
        # 混合正規分布の5%分位点（簡易近似）
        # 各成分の5%分位点を重み付きで平均
        percentiles = []
        for i in range(len(mus)):
            if sigmas[i] > 0:
                p = normal_ppf(0.05, mus[i], sigmas[i])
                percentiles.append(p * weights[i])
        
        if percentiles:
            return sum(percentiles)
        return min(params['mus'])
    
    elif model_name == 'mixture_truncated_normal':
        mus, sigmas, weights = params['mus'], params['sigmas'], params['weights']
        truncation_point = 235
        
        # 混合切断正規分布の5%分位点（簡易近似）
        percentiles = []
        for i in range(len(mus)):
            if sigmas[i] > 0:
                z = (truncation_point - mus[i]) / sigmas[i]
                if z < 10:
                    norm_factor = 1 - normal_cdf(truncation_point, mus[i], sigmas[i])
                    if norm_factor > 1e-10:
                        p_adjusted = 0.05 * norm_factor
                        p = normal_ppf(p_adjusted, mus[i], sigmas[i])
                        percentiles.append(p * weights[i])
        
        if percentiles:
            return sum(percentiles)
        return truncation_point
    
    return None

def main():
    print("各分布モデルの5%分位点（下側下限値）を計算します...")
    
    # データ生成
    data = generate_mixture_truncated_normal(135)
    print(f"生成されたデータ数: {len(data)}")
    print(f"データの範囲: {min(data):.2f} - {max(data):.2f}")
    print(f"平均: {statistics.mean(data):.2f}, 標準偏差: {statistics.stdev(data):.2f}")
    
    # 実際のデータの5%分位点
    sorted_data = sorted(data)
    actual_5th_percentile = sorted_data[int(0.05 * len(data))]
    print(f"\n実際のデータの5%分位点: {actual_5th_percentile:.2f} MPa")
    
    # 各モデルのフィッティング
    print("\n各モデルのフィッティング中...")
    
    # 単一分布モデル
    single_models = fit_single_distributions(data)
    
    # 混合分布モデル
    mixture_models = fit_mixture_models(data)
    
    # 全モデルを結合
    all_models = {**single_models, **mixture_models}
    
    # 5%分位点の計算
    print("\n=== 各モデルの5%分位点（下側下限値） ===")
    print("Model,5th_Percentile_MPa,Parameters,Type")
    
    results = []
    
    for name, model in all_models.items():
        percentile_5 = calculate_5th_percentile(name, model['params'])
        
        if percentile_5 is not None:
            results.append({
                'name': name,
                'percentile_5': percentile_5,
                'n_params': model['n_params'],
                'type': model['type']
            })
            
            print(f"{name},{percentile_5:.2f},{model['n_params']},{model['type']}")
        else:
            print(f"{name},計算不可,{model['n_params']},{model['type']}")
    
    # 結果の分析
    print(f"\n=== 5%分位点の分析 ===")
    
    # 実際の値との比較
    valid_results = [r for r in results if r['percentile_5'] is not None]
    
    if valid_results:
        # 実際の値に最も近いモデル
        closest_model = min(valid_results, key=lambda x: abs(x['percentile_5'] - actual_5th_percentile))
        print(f"実際の5%分位点に最も近いモデル: {closest_model['name']}")
        print(f"  予測値: {closest_model['percentile_5']:.2f} MPa")
        print(f"  実際値: {actual_5th_percentile:.2f} MPa")
        print(f"  誤差: {abs(closest_model['percentile_5'] - actual_5th_percentile):.2f} MPa")
        
        # 全モデルの予測値の範囲
        percentiles = [r['percentile_5'] for r in valid_results]
        print(f"\n全モデルの5%分位点予測範囲:")
        print(f"  最小値: {min(percentiles):.2f} MPa")
        print(f"  最大値: {max(percentiles):.2f} MPa")
        print(f"  範囲: {max(percentiles) - min(percentiles):.2f} MPa")
        
        # 実際の値との誤差
        errors = [abs(r['percentile_5'] - actual_5th_percentile) for r in valid_results]
        print(f"\n実際の値との誤差:")
        print(f"  平均誤差: {statistics.mean(errors):.2f} MPa")
        print(f"  最小誤差: {min(errors):.2f} MPa")
        print(f"  最大誤差: {max(errors):.2f} MPa")
    
    # 詳細な結果をCSVに保存
    print(f"\n=== 詳細結果（CSV形式） ===")
    print("Model,5th_Percentile_MPa,Parameters,Type,Error_vs_Actual")
    for result in results:
        if result['percentile_5'] is not None:
            error = abs(result['percentile_5'] - actual_5th_percentile)
            print(f"{result['name']},{result['percentile_5']:.2f},{result['n_params']},{result['type']},{error:.2f}")
        else:
            print(f"{result['name']},計算不可,{result['n_params']},{result['type']},N/A")

if __name__ == "__main__":
    main()