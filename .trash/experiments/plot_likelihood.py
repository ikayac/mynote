import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.family'] = 'DejaVu Sans'

# 図1: 指数分布の確率密度関数と生存関数
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

t = np.linspace(0, 6, 1000)
lambda_val = 0.3

# 確率密度関数 f(t) = λe^(-λt)
pdf = lambda_val * np.exp(-lambda_val * t)
# 生存関数 S(t) = e^(-λt)  
survival = np.exp(-lambda_val * t)

# 左のグラフ: 確率密度関数
ax1.plot(t, pdf, 'b-', linewidth=2, label=f'f(t) = λe^(-λt), λ={lambda_val}')
ax1.axvline(x=2, color='red', linestyle='--', alpha=0.7)
ax1.fill_between([1.8, 2.2], [0, 0], [lambda_val * np.exp(-lambda_val * 1.8), 
                 lambda_val * np.exp(-lambda_val * 2.2)], alpha=0.3, color='red')
ax1.set_xlabel('時間 t')
ax1.set_ylabel('確率密度')
ax1.set_title('観測1: t₁=2で事象発生\n確率密度 f(2) = λe^(-λ×2)')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.text(2.1, 0.15, 'この面積が\n観測確率', fontsize=10, color='red')

# 右のグラフ: 生存関数
ax2.plot(t, survival, 'g-', linewidth=2, label=f'S(t) = e^(-λt), λ={lambda_val}')
ax2.axvline(x=3, color='red', linestyle='--', alpha=0.7)
ax2.fill_between([0, 3], [0, 0], [1, np.exp(-lambda_val * 3)], alpha=0.3, color='green')
ax2.set_xlabel('時間 t')
ax2.set_ylabel('生存確率')
ax2.set_title('観測2: t₂=3まで事象未発生\n生存確率 S(3) = e^(-λ×3)')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.text(1.5, 0.6, f'S(3) = {np.exp(-lambda_val * 3):.3f}', fontsize=10, color='green')

plt.tight_layout()
plt.savefig('likelihood_components.png', dpi=300, bbox_inches='tight')
plt.show()

# 図2: 尤度関数の形状
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

lambda_range = np.linspace(0.01, 1, 1000)
t1, t2 = 2, 3

# 尤度関数 L(λ) = λe^(-λt₁) × e^(-λt₂)
likelihood = lambda_range * np.exp(-lambda_range * t1) * np.exp(-lambda_range * t2)
log_likelihood = np.log(likelihood)

# 最尤推定値
mle = 1 / (t1 + t2)

ax.plot(lambda_range, likelihood, 'b-', linewidth=2, label='尤度 L(λ)')
ax.axvline(x=mle, color='red', linestyle='--', alpha=0.7, 
           label=f'最尤推定値 λ̂ = {mle:.3f}')
ax.set_xlabel('λ (ハザード率)')
ax.set_ylabel('尤度 L(λ)')
ax.set_title('尤度関数 L(λ) = λe^(-λt₁) × e^(-λt₂)')
ax.grid(True, alpha=0.3)
ax.legend()

# 最大値をマーク
max_likelihood = mle * np.exp(-mle * (t1 + t2))
ax.plot(mle, max_likelihood, 'ro', markersize=8)
ax.text(mle + 0.05, max_likelihood, f'最大尤度\n({mle:.3f}, {max_likelihood:.4f})', 
        fontsize=10, color='red')

plt.tight_layout()
plt.savefig('likelihood_function.png', dpi=300, bbox_inches='tight')
plt.show()

print("図の説明:")
print("図1: 尤度の構成要素")
print("- 左: 観測1の確率密度（事象発生時の確率）")
print("- 右: 観測2の生存確率（事象未発生の確率）")
print("\n図2: 尤度関数")
print("- 横軸: パラメータλ")
print("- 縦軸: 尤度L(λ)")
print("- 最尤推定値で尤度が最大になる")