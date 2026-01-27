#!/usr/bin/env python3
"""
ランダム効果のパラメータ推定方法を実行するスクリプト

このファイルを実行すると、直接観測できないランダム効果を
どのようにパラメータ推定するかが詳しく説明されます。
"""

from parameter_estimation_methods import ParameterEstimationMethods

def main():
    """
    メイン関数
    """
    print("🔍 ランダム効果のパラメータ推定方法を実行します 🔍")
    print("=" * 70)
    
    # 例のインスタンスを作成
    example = ParameterEstimationMethods()
    
    # サンプルデータの作成
    print("\n1. サンプルデータの作成...")
    data = example.create_simple_data(n_subjects=20, n_obs_per_subject=10)
    print("データ作成完了！")
    
    # データの概要表示
    print("\n2. データの概要...")
    print(f"   被験者数: {data['subject_id'].nunique()}")
    print(f"   総観測数: {len(data)}")
    print(f"   時間範囲: {data['time'].min():.1f} ～ {data['time'].max():.1f}")
    print(f"   応答変数範囲: {data['y'].min()} ～ {data['y'].max()}")
    print(f"   平均応答変数: {data['y'].mean():.2f}")
    
    # 推定問題の本質
    print("\n3. 推定問題の本質...")
    example.explain_estimation_problem()
    
    # ラプラス近似法による最尤推定
    print("\n4. ラプラス近似法による最尤推定...")
    example.demonstrate_laplace_approximation()
    
    # MCMC法によるベイズ推定
    print("\n5. MCMC法によるベイズ推定...")
    example.demonstrate_mcmc_method()
    
    # 推定方法の比較
    print("\n6. 推定方法の比較...")
    example.compare_estimation_methods()
    
    # 実用的な実装方法
    print("\n7. 実用的な実装方法...")
    example.show_practical_implementation()
    
    # 収束診断の方法
    print("\n8. 収束診断の方法...")
    example.demonstrate_convergence_diagnostics()
    
    print("\n" + "=" * 70)
    print("✅ 実行完了！")
    print("\n💡 さらに詳しい数学的説明は 'estimation_math_details.md' を参照してください。")
    print("🎯 釣果尾数のGLMM例は 'fishing_glmm_example.py' を実行してください。")
    print("📚 基本的なGLMMの理解は 'glmm_tutorial.py' を実行してください。")

if __name__ == "__main__":
    main()