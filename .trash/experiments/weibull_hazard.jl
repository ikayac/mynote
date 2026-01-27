using Plots
using LaTeXStrings

# ワイブルハザードモデルの定義
# S(x) = exp(-θ * x^m)
# λ(x) = θ * m * x^(m-1)

function survival_function(x, theta, m)
    return exp(-theta * x^m)
end

function hazard_function(x, theta, m)
    # x=0 で m<1 の場合、ゼロ除算になるため微小値を加えるか分岐処理
    if x == 0 && m < 1
        return Inf
    end
    return theta * m * x^(m-1)
end

# パラメータ設定
theta = 0.01  # スケールパラメータのようなもの（適当な値）
x_range = 0:0.1:20

# プロットの準備（形状パラメータ m を変化させる）
ms = [0.5, 1.0, 2.0, 4.0]
labels = ["m=0.5 (初期)" "m=1.0 (偶発)" "m=2.0 (摩耗)" "m=4.0 (摩耗)"]

# ハザード関数のプロット
p1 = plot(title="Hazard Function λ(x)", xlabel="Time x", ylabel="Hazard Rate")
for (i, m) in enumerate(ms)
    y = [hazard_function(x, theta, m) for x in x_range]
    # y軸が見やすくなるように制限
    plot!(p1, x_range, y, label=labels[i], ylims=(0, 0.15), lw=2)
end

# 生存関数のプロット
p2 = plot(title="Survival Function S(x)", xlabel="Time x", ylabel="Survival Prob")
for (i, m) in enumerate(ms)
    y = [survival_function(x, theta, m) for x in x_range]
    plot!(p2, x_range, y, label=labels[i], lw=2)
end

# まとめて表示
plot(p1, p2, layout=(2, 1), size=(800, 800))

# 保存
if !ispath("study/experiments/fig")
    mkpath("study/experiments/fig")
end
savefig("study/experiments/fig/weibull_plots.png")
println("プロットを study/experiments/fig/weibull_plots.png に保存しました。")
