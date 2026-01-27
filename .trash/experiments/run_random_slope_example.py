#!/usr/bin/env python3
"""
ランダム傾きモデルの例を実行するスクリプト

このファイルを実行すると、切片だけでなく係数にもランダム効果を入れる
ランダム傾きモデルの例が表示されます。
"""

from fishing_random_slope_example import FishingRandomSlopeGLMM

def main():
    """
    メイン関数
    """
    print("🎣 釣果尾数のランダム傾きGLMM例を実行します 🎣")
    print("=" * 70)
    
    # 例のインスタンスを作成
    example = FishingRandomSlopeGLMM()
    
    # サンプルデータの作成
    print("\n📊 1. サンプルデータの作成...")
    data = example.create_random_slope_data(n_anglers=20, n_trips=10)
    print(f"   作成完了！ {len(data)}件のデータ")
    print(f"   釣り人数: {data['angler_id'].nunique()}")
    print(f"   釣行回数: {data['trip'].nunique()}")
    
    # データの概要表示
    print("\n📈 2. データの概要...")
    print(f"   気温範囲: {data['temperature'].min():.1f}度 ～ {data['temperature'].max():.1f}度")
    print(f"   釣果尾数範囲: {data['catch_count'].min()}尾 ～ {data['catch_count'].max()}尾")
    print(f"   平均釣果尾数: {data['catch_count'].mean():.2f}尾")
    
    # ランダム傾きモデルの構造説明
    print("\n🔍 3. ランダム傾きモデルの構造説明...")
    example.explain_random_slope_model()
    
    # 個人差の具体的な例
    print("\n👥 4. 個人差の具体的な例...")
    example.demonstrate_individual_differences()
    
    # 手計算による具体例
    print("\n🧮 5. 手計算による具体例...")
    example.show_hand_calculation_examples()
    
    # データの可視化
    print("\n📊 6. データの可視化...")
    example.visualize_random_slope_model()
    
    # モデルの比較
    print("\n⚖️ 7. モデルの比較...")
    example.compare_models()
    
    # 尤度関数の構造
    print("\n📚 8. 尤度関数の構造...")
    example.demonstrate_likelihood_structure()
    
    print("\n" + "=" * 70)
    print("✅ 実行完了！")
    print("\n💡 さらに詳しい数学的説明は 'random_slope_math_explanation.md' を参照してください。")
    print("🎯 基本的なランダム切片モデルは 'fishing_glmm_example.py' を実行してください。")
    print("📖 基本的なGLMMの理解は 'glmm_tutorial.py' を実行してください。")

if __name__ == "__main__":
    main()