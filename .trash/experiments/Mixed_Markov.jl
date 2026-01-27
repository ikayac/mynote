using DelimitedFiles
using Random
using Statistics
using SpecialFunctions
using Distributions
using Printf
using Dates

# 尤度関数
function Likelihood_mix(N, K, Lk, Before, After, Interval, Expla, Beta, Epsilon, Max_val)
    Theta = exp.(Beta * Expla')
    Pi = zeros(N)
    n = 0
    for k in 1:K
        Eps = Epsilon[k]
        for lk in 1:Lk[k]
            n += 1
            Be = Int(Before[n])
            Af = Int(After[n])
            Intvl = Interval[n]

            if Af == Max_val
                if Be == Max_val
                    Pi[n] = 1.0
                else
                    pa = 0.0
                    for j in Be:(Af-1)
                        p0 = 0.0
                        for s in Be:j
                            p1 = 1.0
                            p2 = 1.0
                            if s != Be
                                for m in Be:(s-1)
                                    p1 *= Theta[m, n] / (Theta[m, n] - Theta[s, n])
                                end
                            end
                            if s != j
                                for m in s:(j-1)
                                    p2 *= Theta[m, n] / (Theta[m+1, n] - Theta[s, n])
                                end
                            end
                            p3 = exp(-Theta[s, n] * Eps * Intvl)
                            p0 += p1 * p2 * p3
                        end
                        pa += p0
                    end
                    Pi[n] = 1.0 - pa
                end
            else
                for s in Be:Af
                    p1 = 1.0
                    p2 = 1.0
                    if s != Be
                        for m in Be:(s-1)
                            p1 *= Theta[m, n] / (Theta[m, n] - Theta[s, n])
                        end
                    end
                    if s != Af
                        for m in s:(Af-1)
                            p2 *= Theta[m, n] / (Theta[m+1, n] - Theta[s, n])
                        end
                    end
                    p3 = exp(-Theta[s, n] * Eps * Intvl)
                    Pi[n] += p1 * p2 * p3
                end
            end
            if Pi[n] <= 0
                Pi[n] = 1e-100
            end
        end
    end
    Ln_Pi = log.(Pi)
    return sum(Ln_Pi)
end

function Likelihood_eps(k, Lk, Before, After, Interval, Theta, Eps, Max_val)
    num_samples = Lk[k]
    Pi = zeros(num_samples)

    count1 = (k == 1) ? 0 : sum(Lk[1:k-1])
    count2 = 0

    for lk in 1:num_samples
        count1 += 1
        count2 += 1
        Be = Int(Before[count1])
        Af = Int(After[count1])
        Intvl = Interval[count1]

        if Af == Max_val
            if Be == Max_val
                Pi[count2] = 1.0
            else
                pa = 0.0
                for j in Be:(Af-1)
                    p0 = 0.0
                    for s in Be:j
                        p1 = 1.0
                        p2 = 1.0
                        if s != Be
                            for m in Be:(s-1)
                                p1 *= Theta[m, count1] / (Theta[m, count1] - Theta[s, count1])
                            end
                        end
                        if s != j
                            for m in s:(j-1)
                                p2 *= Theta[m, count1] / (Theta[m+1, count1] - Theta[s, count1])
                            end
                        end
                        p3 = exp(-Theta[s, count1] * Eps * Intvl)
                        p0 += p1 * p2 * p3
                    end
                    pa += p0
                end
                Pi[count2] = 1.0 - pa
            end
        else
            for s in Be:Af
                p1 = 1.0
                p2 = 1.0
                if s != Be
                    for m in Be:(s-1)
                        p1 *= Theta[m, count1] / (Theta[m, count1] - Theta[s, count1])
                    end
                end
                if s != Af
                    for m in s:(Af-1)
                        p2 *= Theta[m, count1] / (Theta[m+1, count1] - Theta[s, count1])
                    end
                end
                p3 = exp(-Theta[s, count1] * Eps * Intvl)
                Pi[count2] += p1 * p2 * p3
            end
        end
        if Pi[count2] <= 0
             Pi[count2] = 1e-100
        end
    end
    Ln_Pi = log.(Pi)
    return sum(Ln_Pi)
end

function Geweke(Sample)
    Dim = ndims(Sample)
    Size = size(Sample, Dim)

    N1 = Int(floor(0.1 * Size))
    N2 = Int(floor(0.5 * Size))
    q = 20

    if Dim == 2
        S1 = Sample[:, 1:N1]
        S2 = Sample[:, Size-N2+1:Size]

        Average1 = mean(S1, dims=2)
        Average2 = mean(S2, dims=2)

        Omega01 = var(S1, dims=2)
        Omega02 = var(S2, dims=2)

        Omega_omega1 = zeros(size(S1, 1), 1)
        Omega_omega2 = zeros(size(S2, 1), 1)

        for s in 1:q
            Omega1 = zeros(size(S1, 1), 1)
            for g in (s+1):N1
                Omega1 .+= (S1[:, g] .- Average1) .* (S1[:, g-s] .- Average1)
            end
            Omega1 ./= N1

            Omega2 = zeros(size(S2, 1), 1)
            for g in (Size-N2+s+1):Size
                idx = g - (Size-N2)
                idx_s = idx - s
                if idx_s > 0
                    Omega2 .+= (S2[:, idx] .- Average2) .* (S2[:, idx_s] .- Average2)
                end
            end
            Omega2 ./= N2

            Omega = 1.0 - s / (q + 1)
            Omega_omega1 .+= Omega1 .* Omega
            Omega_omega2 .+= Omega2 .* Omega
        end

        F1 = Omega01 .+ Omega_omega1 .* 2
        F2 = Omega02 .+ Omega_omega2 .* 2

        Nu1 = F1 ./ N1
        Nu2 = F2 ./ N2

        return (Average1 .- Average2) ./ sqrt.(Nu1 .+ Nu2)

    elseif Dim == 3
        S1 = Sample[:, :, 1:N1]
        S2 = Sample[:, :, Size-N2+1:Size]

        Average1 = mean(S1, dims=3)
        Average2 = mean(S2, dims=3)

        Omega01 = var(S1, dims=3)
        Omega02 = var(S2, dims=3)

        Omega_omega1 = zeros(size(S1, 1), size(S1, 2), 1)
        Omega_omega2 = zeros(size(S2, 1), size(S2, 2), 1)

         for s in 1:q
            Omega1 = zeros(size(S1, 1), size(S1, 2), 1)
            for g in (s+1):N1
                Omega1 .+= (S1[:, :, g] .- Average1) .* (S1[:, :, g-s] .- Average1)
            end
            Omega1 ./= N1

            Omega2 = zeros(size(S2, 1), size(S2, 2), 1)
            for g in (Size-N2+s+1):Size
                idx = g - (Size-N2)
                idx_s = idx - s
                if idx_s > 0
                    Omega2 .+= (S2[:, :, idx] .- Average2) .* (S2[:, :, idx_s] .- Average2)
                end
            end
            Omega2 ./= N2

            Omega = 1.0 - s / (q + 1)
            Omega_omega1 .+= Omega1 .* Omega
            Omega_omega2 .+= Omega2 .* Omega
        end

        F1 = Omega01 .+ Omega_omega1 .* 2
        F2 = Omega02 .+ Omega_omega2 .* 2

        Nu1 = F1 ./ N1
        Nu2 = F2 ./ N2

        return (Average1 .- Average2) ./ sqrt.(Nu1 .+ Nu2)

    elseif Dim == 1
        S1 = Sample[1:N1]
        S2 = Sample[Size-N2+1:Size]

        Average1 = mean(S1)
        Average2 = mean(S2)

        Omega01 = var(S1)
        Omega02 = var(S2)

        Omega_omega1 = 0.0
        Omega_omega2 = 0.0

        for s in 1:q
            Omega1 = 0.0
            for g in (s+1):N1
                Omega1 += (S1[g] - Average1) * (S1[g-s] - Average1)
            end
            Omega1 /= N1

            Omega2 = 0.0
            for g in (Size-N2+s+1):Size
                idx = g - (Size-N2)
                idx_s = idx - s
                if idx_s > 0
                    Omega2 += (S2[idx] - Average2) * (S2[idx_s] - Average2)
                end
            end
            Omega2 /= N2

            Omega = 1.0 - s / (q + 1)
            Omega_omega1 += Omega1 * Omega
            Omega_omega2 += Omega2 * Omega
        end

        F1 = Omega01 + Omega_omega1 * 2
        F2 = Omega02 + Omega_omega2 * 2

        Nu1 = F1 / N1
        Nu2 = F2 / N2

        return (Average1 - Average2) / sqrt(Nu1 + Nu2)
    end
end

function main()
    println("処理開始")

    # データの読み込み
    # Check if header exists by trying to read it
    raw_data, header = readdlm("Data_sampled.txt", '\t', header=true)
    # If the first column is string, it's likely a header. If number, maybe no header or skipped?
    # Actually, the file content provided shows a header line: "事前健全度　事後健全度..."

    data = Float64.(raw_data)
    N = size(data, 1)

    Before = data[:, 1]
    After = data[:, 2]
    Interval = data[:, 3]
    Group = Int.(data[:, 4])
    Expla = hcat(ones(N), data[:, 5:6])

    I_max = Int(maximum(After))
    M = size(Expla, 2)
    K = maximum(Group)
    Lk = [count(x -> x == k, Group) for k in 1:K]

    println("データ読み込み完了: N=$N, I=$I_max, M=$M, K=$K")

    Loop = 12000
    Burnin = 2000

    Ave = 1.0
    Dis = 0.1

    Width_beta = ones(I_max-1, M) ./ 10.0

    Skip = [1 0 1;
            1 0 1;
            1 1 0;
            1 1 0;
            1 0 0;
            1 1 0]

    if size(Skip, 1) < I_max - 1
        println("Skip matrix is smaller than I-1. Padding with ones.")
        padding = ones(Int, I_max - 1 - size(Skip, 1), M)
        Skip = vcat(Skip, padding)
    elseif size(Skip, 1) > I_max - 1
        println("Skip matrix is larger than I-1. Trimming.")
        Skip = Skip[1:I_max-1, :]
    end

    Record_beta = zeros(I_max-1, M, Loop)
    Record_eps = zeros(K, Loop)
    Record_phi = zeros(Loop)

    Record_beta[:, :, 1] = -rand(I_max-1, M) .* Skip
    Record_eps[:, 1] .= 1.0
    Record_phi[1] = rand(Gamma(Ave^2/Dis, Dis/Ave))

    current_LH_mix = Likelihood_mix(N, K, Lk, Before, After, Interval, Expla, Record_beta[:, :, 1], Record_eps[:, 1], I_max)

    Theta = exp.(Record_beta[:, :, 1] * Expla')
    current_LH_eps = zeros(K)
    for k in 1:K
        current_LH_eps[k] = Likelihood_eps(k, Lk, Before, After, Interval, Theta, Record_eps[k, 1], I_max)
    end

    dist_phi = Gamma(Record_phi[1], 1.0/Record_phi[1])
    current_LH_phi = sum(log.(pdf.(dist_phi, Record_eps[:, 1])))

    println("サンプリング開始")
    time_start = time()

    accept_beta = zeros(I_max-1, M)
    accept_eps = zeros(K)
    accept_phi = 0

    for loop in 2:Loop
        Record_beta[:, :, loop] = Record_beta[:, :, loop-1]
        Record_eps[:, loop] = Record_eps[:, loop-1]
        Record_phi[loop] = Record_phi[loop-1]

        # --- Beta Sampling ---
        for m in 1:M
            for i in 1:(I_max-1)
                if Skip[i, m] == 0
                    continue
                end

                Step = randn() * Width_beta[i, m]
                Newbeta_im = Record_beta[i, m, loop] + Step

                Newbeta = copy(Record_beta[:, :, loop])
                Newbeta[i, m] = Newbeta_im

                LH_Newbeta = Likelihood_mix(N, K, Lk, Before, After, Interval, Expla, Newbeta, Record_eps[:, loop], I_max)

                r = exp(LH_Newbeta - current_LH_mix)

                if rand() <= min(1.0, r)
                    Record_beta[i, m, loop] = Newbeta_im
                    current_LH_mix = LH_Newbeta
                    accept_beta[i, m] += 1
                end
            end
        end

        # --- Epsilon Sampling ---
        Theta = exp.(Record_beta[:, :, loop] * Expla')

        for k in 1:K
            Neweps = rand(Gamma(Record_phi[loop], 1.0/Record_phi[loop]))
            LH_Neweps = Likelihood_eps(k, Lk, Before, After, Interval, Theta, Neweps, I_max)

            r = exp(LH_Neweps - current_LH_eps[k])

            if rand() <= min(1.0, r)
                Record_eps[k, loop] = Neweps
                current_LH_eps[k] = LH_Neweps
                accept_eps[k] += 1
            end
        end

        # --- Phi Sampling ---
        dist_phi_curr = Gamma(Record_phi[loop], 1.0/Record_phi[loop])
        LH_phi_curr = sum(log.(pdf.(dist_phi_curr, Record_eps[:, loop])))

        Newphi = rand(Gamma(Ave^2/Dis, Dis/Ave))
        dist_phi_new = Gamma(Newphi, 1.0/Newphi)
        LH_phi_new = sum(log.(pdf.(dist_phi_new, Record_eps[:, loop])))

        r = exp(LH_phi_new - LH_phi_curr)

        if rand() <= min(1.0, r)
            Record_phi[loop] = Newphi
            accept_phi += 1
        end

        if loop % (Loop ÷ 10) == 0
            elapsed = time() - time_start
            println("サンプリング $loop 回完了, 経過: $(round(elapsed, digits=1))秒")
        end
    end

    println("集計中...")
    Sample_beta = Record_beta[:, :, (Burnin+1):Loop]
    Sample_eps = Record_eps[:, (Burnin+1):Loop]
    Sample_phi = Record_phi[(Burnin+1):Loop]

    Expected_beta = dropdims(mean(Sample_beta, dims=3), dims=3)
    Expected_eps = dropdims(mean(Sample_eps, dims=2), dims=2)
    Expected_phi = mean(Sample_phi)

    println("--- 結果 ---")
    println("Expected Beta:")
    display(Expected_beta)
    println("Expected Phi: $Expected_phi")
    println("Geweke Beta:")
    display(Geweke(Sample_beta))

    writedlm("Result_Beta.txt", Expected_beta)
    writedlm("Result_Eps.txt", Expected_eps)
    writedlm("Result_Phi.txt", [Expected_phi])

    println("結果を保存しました。")

    Theta_ave = exp.(Expected_beta * mean(Expla, dims=1)')
    ET_ave = 1.0 ./ Theta_ave
    Acc_ET_ave = vcat(0.0, cumsum(ET_ave, dims=1))
    writedlm("Result_DegradationPath.txt", Acc_ET_ave)
    println("劣化パス(平均)を保存しました。")

end

main()
