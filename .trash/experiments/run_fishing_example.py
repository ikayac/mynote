#!/usr/bin/env python3
"""
釣果尾数のGLMM例を実行するスクリプト

このファイルを実行すると、釣果尾数のGLMMの例が表示されます。
"""

from fishing_glmm_example import FishingGLMMExample

def main():
    """
    メイン関数
    """
    print("🎣 釣果尾数のGLMM例を実行します 🎣")
    print("=" * 60)
    
    # 例のインスタンスを作成
    example = FishingGLMMExample()
    
    # サンプルデータの作成
    print("\n📊 1. サンプルデータの作成...")
    data = example.create_fishing_data(n_anglers=15, n_trips=8)
    print(f"   作成完了！ {len(data)}件のデータ")
    print(f"   釣り人数: {data['angler_id'].nunique()}")
    print(f"   釣行回数: {data['trip'].nunique()}")
    
    # データの概要表示
    print("\n📈 2. データの概要...")
    print(f"   気温範囲: {data['temperature'].min():.1f}度 ～ {data['temperature'].max():.1f}度")
    print(f"   釣果尾数範囲: {data['catch_count'].min()}尾 ～ {data['catch_count'].max()}尾")
    print(f"   平均釣果尾数: {data['catch_count'].mean():.2f}尾")
    
    # GLMMの構造説明
    print("\n🔍 3. GLMMの構造説明...")
    example.explain_glmm_structure()
    
    # 手計算による説明
    print("\n🧮 4. 手計算による説明...")
    example.demonstrate_hand_calculation()
    
    # パラメータの解釈
    print("\n📝 5. パラメータの解釈...")
    example.show_parameter_interpretation()
    
    # データの可視化
    print("\n📊 6. データの可視化...")
    example.show_data_visualization()
    
    # モデルの比較
    print("\n⚖️ 7. モデルの比較...")
    example.demonstrate_model_comparison()
    
    print("\n" + "=" * 60)
    print("✅ 実行完了！")
    print("\n💡 さらに詳しい数学的説明は 'fishing_glmm_math_explanation.md' を参照してください。")
    print("📚 基本的なGLMMの理解は 'glmm_tutorial.py' を実行してください。")

if __name__ == "__main__":
    main()