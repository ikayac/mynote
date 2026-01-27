# 統計的モデリング手法の詳細説明

## 📊 概要

このドキュメントでは、強度分布の統計的モデリングで使用される手法について詳細に説明します。

## 🔬 データ生成手法

### 混合切断正規分布の生成

#### 1. 基本パラメータ
```python
# 第1成分（270MPa付近）
mu1, sigma1 = 270, 15
# 第2成分（300MPa付近）  
mu2, sigma2 = 300, 12
# 混合比率
weights = [0.6, 0.4]
# 切断点
truncation_point = 235
```

#### 2. Box-Muller変換
正規分布の乱数を生成するために、Box-Muller変換を使用：

```python
def box_muller(mu, sigma):
    u1 = random.random()  # 一様乱数
    u2 = random.random()  # 一様乱数
    
    # Box-Muller変換
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    
    return mu + sigma * z0
```

#### 3. 切断処理
切断点以下の値を除外し、適切なサンプル数を確保：

```python
# 第1成分からサンプリング
samples1 = []
while len(samples1) < n1:
    sample = box_muller(mu1, sigma1)
    if sample >= truncation_point:
        samples1.append(sample)
```

## 📈 分布フィッティング手法

### 1. 単一分布のフィッティング

#### 正規分布
```python
# 最尤推定によるパラメータ推定
mu = np.mean(data)
sigma = np.std(data)
```

#### ワイブル分布
```python
# scipy.stats.weibull_min.fit()を使用
shape, loc, scale = stats.weibull_min.fit(data)
```

#### 対数正規分布
```python
# 対数変換後のデータで正規分布フィッティング
log_data = np.log(data)
mu_log = np.mean(log_data)
sigma_log = np.std(log_data)
```

#### ガンマ分布
```python
# モーメント法による初期値推定
if std > 0:
    shape_gamma = (mean - min_val) ** 2 / (std ** 2)
    scale_gamma = (std ** 2) / (mean - min_val)
```

### 2. 混合分布のフィッティング

#### 混合正規分布
```python
# GaussianMixtureを使用
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data.reshape(-1, 1))

# パラメータの抽出
mus = gmm.means_.flatten()
sigmas = np.sqrt(gmm.covariances_.flatten())
weights = gmm.weights_
```

#### 混合切断正規分布
```python
# 簡易的なクラスタリングによる初期値推定
sorted_data = np.sort(data)
n = len(data)
group_size = n // 2

group1 = sorted_data[:group_size]
group2 = sorted_data[group_size:]

mu1, sigma1 = np.mean(group1), np.std(group1)
mu2, sigma2 = np.mean(group2), np.std(group2)
weight1, weight2 = len(group1) / n, len(group2) / n
```

## 📊 統計的評価手法

### 1. 対数尤度の計算

#### 正規分布
```python
def normal_log_likelihood(data, mu, sigma):
    return np.sum(stats.norm.logpdf(data, mu, sigma))
```

#### 切断正規分布
```python
def truncated_normal_log_likelihood(data, mu, sigma, truncation_point):
    # 切断正規分布の対数尤度
    log_pdf = stats.norm.logpdf(data, mu, sigma)
    log_cdf_trunc = stats.norm.logcdf(truncation_point, mu, sigma)
    log_surv_trunc = np.log(1 - np.exp(log_cdf_trunc))
    
    return np.sum(log_pdf - log_surv_trunc)
```

#### 混合分布
```python
def mixture_log_likelihood(data, components, weights):
    log_likelihood = 0
    for i, component in enumerate(components):
        log_likelihood += weights[i] * component.pdf(data)
    
    return np.sum(np.log(log_likelihood + 1e-10))
```

### 2. AICの計算

```python
def calculate_aic(log_likelihood, n_params, n_samples):
    """
    Akaike Information Criterion (AIC)の計算
    
    AIC = 2k - 2ln(L)
    
    パラメータ:
    - log_likelihood: 対数尤度
    - n_params: パラメータ数
    - n_samples: サンプル数
    
    戻り値:
    - AIC値（小さいほど良い）
    """
    return 2 * n_params - 2 * log_likelihood
```

### 3. 5%分位点の計算

#### 正規分布
```python
def normal_5th_percentile(mu, sigma):
    return stats.norm.ppf(0.05, mu, sigma)
```

#### 切断正規分布
```python
def truncated_normal_5th_percentile(mu, sigma, truncation_point):
    # 切断点以下の確率を考慮して調整
    z = (truncation_point - mu) / sigma
    if z < 10:
        norm_factor = 1 - stats.norm.cdf(truncation_point, mu, sigma)
        if norm_factor > 1e-10:
            p_adjusted = 0.05 * norm_factor
            return stats.norm.ppf(p_adjusted, mu, sigma)
    
    return truncation_point
```

#### ワイブル分布
```python
def weibull_5th_percentile(shape, loc, scale):
    return loc + scale * ((-np.log(0.95)) ** (1 / shape))
```

## 🎨 可視化手法

### 1. ヒストグラムの作成

```python
def create_histogram(data, bins=20):
    """
    データからヒストグラムを作成
    
    パラメータ:
    - data: 分析対象データ
    - bins: ビン数
    
    戻り値:
    - bin_centers: ビンの中心値
    - bin_counts: 各ビンの度数
    """
    min_val = np.min(data)
    max_val = np.max(data)
    bin_width = (max_val - min_val) / bins
    
    bin_centers = []
    for i in range(bins):
        center = min_val + (i + 0.5) * bin_width
        bin_centers.append(center)
    
    # ヒストグラムの作成
    hist, _ = np.histogram(data, bins=bins)
    
    return bin_centers, hist
```

### 2. 密度曲線の描画

```python
def plot_density_curves(x_range, models, results):
    """
    各モデルの密度曲線を描画
    
    パラメータ:
    - x_range: x軸の範囲
    - models: フィッティングされたモデル
    - results: 分析結果
    """
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        if name in results:
            try:
                # 各モデルに応じた密度曲線の描画
                if name == 'single_normal':
                    mu, sigma = model['params']['mu'], model['params']['sigma']
                    plt.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 
                            color=colors[i], linewidth=2, label=name, alpha=0.8)
                
                # 他のモデルも同様に...
                
            except Exception as e:
                print(f"Warning: Could not plot {name}: {e}")
```

### 3. 包括的可視化の構成

```python
def create_comprehensive_visualization(data, all_models, results):
    """
    9つのサブプロットによる包括的可視化
    
    サブプロット構成:
    1. 全体のヒストグラムと密度曲線比較
    2. 270MPa付近の詳細（第1ピーク）
    3. 300MPa付近の詳細（第2ピーク）
    4. 下側尾部の詳細（235-250MPa）
    5. AICによるモデル比較
    6. 5%分位点の比較
    7. 対数尤度の比較
    8. パラメータ数の比較
    9. フィット具合の総合評価
    """
    fig = plt.figure(figsize=(20, 15))
    
    # 各サブプロットの作成
    # ... (詳細な実装)
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig
```

## 🔍 数値的安定性の考慮

### 1. 対数尤度計算での数値的安定性

```python
def stable_log_likelihood(data, model_name, params):
    """
    数値的に安定した対数尤度の計算
    
    問題点:
    - log(0) = -∞
    - 非常に小さい確率での数値的不安定性
    
    対策:
    - 小さな値の追加 (1e-10)
    - 適切な範囲チェック
    """
    try:
        if model_name == 'mixture_normal':
            mus, sigmas, weights = params['mus'], params['sigmas'], params['weights']
            
            log_likelihood = 0
            for i in range(len(mus)):
                if sigmas[i] > 0:
                    component_likelihood = weights[i] * stats.norm.pdf(data, mus[i], sigmas[i])
                    log_likelihood += component_likelihood
            
            # 数値的安定性のための小さな値の追加
            return np.sum(np.log(log_likelihood + 1e-10))
            
    except Exception as e:
        print(f"Error in log likelihood calculation: {e}")
        return -np.inf
```

### 2. パラメータの妥当性チェック

```python
def validate_parameters(params, model_name):
    """
    パラメータの妥当性をチェック
    
    チェック項目:
    - 分散パラメータ > 0
    - 形状パラメータ > 0
    - 重みの合計 = 1
    """
    if model_name == 'single_normal':
        mu, sigma = params['mu'], params['sigma']
        if sigma <= 0:
            return False
    
    elif model_name == 'weibull':
        shape, loc, scale = params['shape'], params['loc'], params['scale']
        if shape <= 0 or scale <= 0:
            return False
    
    elif model_name == 'mixture_normal':
        mus, sigmas, weights = params['mus'], params['sigmas'], params['weights']
        if any(s <= 0 for s in sigmas):
            return False
        if abs(sum(weights) - 1.0) > 1e-6:
            return False
    
    return True
```

## 📚 参考文献

1. Akaike, H. (1974). "A new look at the statistical model identification". IEEE Transactions on Automatic Control, 19(6), 716-723.

2. McLachlan, G., & Peel, D. (2000). "Finite Mixture Models". Wiley Series in Probability and Statistics.

3. Box, G. E. P., & Muller, M. E. (1958). "A Note on the Generation of Random Normal Deviates". The Annals of Mathematical Statistics, 29(2), 610-611.

4. Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994). "Continuous Univariate Distributions, Volume 1". Wiley Series in Probability and Statistics.

## 🔄 今後の改善点

1. **最適化アルゴリズムの改善**: より効率的なパラメータ推定
2. **ベイズ推定の導入**: 不確実性の定量化
3. **交差検証**: モデルの汎化性能の評価
4. **自動モデル選択**: データに基づく最適モデルの自動選択
5. **リアルタイム可視化**: インタラクティブな分析環境