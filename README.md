# 強度分布の統計的モデリング分析

## 📊 プロジェクト概要

このプロジェクトは、材料の強度分布を様々な統計モデルで分析し、最適な分布モデルを特定する包括的な分析ツールです。特に、切断点を持つ混合分布のフィッティングと、AICによるモデル比較に焦点を当てています。

## 🎯 主な特徴

- **混合切断正規分布データの生成**: 270MPaと300MPa付近にピークを持つ現実的なデータ
- **多様な分布モデルのフィッティング**: 単一分布から混合分布まで11種類のモデル
- **統計的モデル選択**: AIC、対数尤度、5%分位点による包括的評価
- **高品質な可視化**: 9つのサブプロットによる詳細な分析結果の可視化
- **Google Colab対応**: クラウド環境での完全実行が可能

## 📁 ファイル構成

```
strength-distribution-analysis/
├── README.md                           # このファイル
├── requirements.txt                    # 必要なPythonパッケージ
├── complete_analysis_colab.py         # Google Colab用完全版スクリプト
├── simple_visualization.py            # 標準ライブラリのみ使用版
├── percentile_analysis.py             # 5%分位点分析専用
├── examples/                          # 実行例とサンプルデータ
│   ├── sample_output.txt             # 実行結果のサンプル
│   └── visualization_example.py      # 可視化の例
└── docs/                             # 詳細なドキュメント
    ├── methodology.md                 # 手法の詳細説明
    └── results_interpretation.md     # 結果の解釈方法
```

## 🚀 クイックスタート

### 1. リポジトリのクローン
```bash
git clone https://github.com/yourusername/strength-distribution-analysis.git
cd strength-distribution-analysis
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 3. 分析の実行
```bash
python complete_analysis_colab.py
```

## 🔧 必要な環境

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- Seaborn

## 📈 分析対象の分布モデル

### 単一分布モデル
1. **単一正規分布** (2パラメータ)
2. **切断正規分布** (2パラメータ)
3. **ワイブル分布** (3パラメータ)
4. **対数正規分布** (3パラメータ)
5. **ガンマ分布** (3パラメータ)
6. **一様分布** (2パラメータ) - "nothing model"

### 混合分布モデル
7. **混合正規分布** (5パラメータ)
8. **混合切断正規分布** (5パラメータ)
9. **混合ワイブル分布** (7パラメータ)
10. **混合ガンマ分布** (7パラメータ)
11. **混合対数正規分布** (7パラメータ)

## 📊 分析結果の例

### AICによるモデルランキング
```
1. mixture_truncated_normal    AIC: 1234.56
2. mixture_normal_2           AIC: 1245.67
3. truncated_normal           AIC: 1256.78
4. single_normal              AIC: 1267.89
5. weibull                    AIC: 1278.90
...
```

### 5%分位点の予測精度
```
実際のデータの5%分位点: 247.91 MPa

gamma: 247.55 MPa (誤差: 0.36 MPa)
weibull: 247.01 MPa (誤差: 0.91 MPa)
mixture_truncated_normal: 248.23 MPa (誤差: 0.32 MPa)
...
```

## 🎨 可視化の特徴

### 9つのサブプロット
1. **全体のヒストグラムと密度曲線比較**
2. **270MPa付近の詳細（第1ピーク）**
3. **300MPa付近の詳細（第2ピーク）**
4. **下側尾部の詳細（235-250MPa）**
5. **AICによるモデル比較**
6. **5%分位点の比較**
7. **対数尤度の比較**
8. **パラメータ数の比較**
9. **フィット具合の総合評価**

## 🔬 手法の詳細

### データ生成
- Box-Muller変換による正規分布の生成
- 切断点（235MPa）での適切なサンプリング
- 2つのピーク（270MPa, 300MPa）の混合比率制御

### モデルフィッティング
- 最尤推定によるパラメータ推定
- 混合モデルのEMアルゴリズム
- 数値的安定性のためのフォールバック機能

### 統計的評価
- Akaike Information Criterion (AIC)
- 対数尤度の比較
- 5%分位点の予測精度評価

## 📚 使用例

### Google Colabでの実行
```python
# 必要なライブラリのインストール
!pip install numpy matplotlib seaborn scipy scikit-learn

# スクリプトの実行
exec(open('complete_analysis_colab.py').read())
```

### カスタムデータでの分析
```python
from complete_analysis_colab import *

# 独自のデータで分析
custom_data = your_strength_data
results = analyze_custom_data(custom_data)
```

## 🤝 貢献方法

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 👥 作者

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 謝辞

- 統計的モデリングの理論的基盤を提供してくださった研究者の方々
- 可視化ライブラリの開発者コミュニティ
- フィードバックと改善提案をいただいたユーザーの方々

## 📞 サポート

質問や問題がございましたら、[Issues](https://github.com/yourusername/strength-distribution-analysis/issues)ページでお知らせください。

## 🔄 更新履歴

- **v1.0.0** - 初期リリース
  - 基本的な分布フィッティング機能
  - AICによるモデル比較
  - 基本的な可視化

- **v1.1.0** - 機能拡張
  - 混合分布モデルの追加
  - 5%分位点の分析
  - 包括的可視化の実装

- **v1.2.0** - Google Colab対応
  - クラウド環境での実行対応
  - エラーハンドリングの改善
  - ドキュメントの充実

---

⭐ このプロジェクトが役に立ったら、スターを付けてください！
# 一般化線形混合モデル（GLMM）学習リソース

このリポジトリは、一般化線形混合モデル（Generalized Linear Mixed Models, GLMM）の理解を深めるための包括的な学習リソースを提供します。

## 📚 学習リソースの構成

### 1. `glmm_tutorial.py` - 基礎チュートリアル
- GLMMの基本概念の説明
- サンプルデータの生成と可視化
- 理論的背景の解説
- 実用的な応用例の紹介

### 2. `glmm_practical_examples.py` - 実践例
- 縦断データの分析例
- 様々な統計パッケージの使用例
- 混合効果モデルの実装
- ベイズGLMMの概念説明

### 3. `glmm_theory.md` - 理論的背景
- 数学的定式化の詳細
- 推定方法の解説
- モデル診断と選択
- 実用的な考慮事項

## 🚀 セットアップと実行

### 必要なパッケージのインストール
```bash
pip install -r requirements.txt
```

### 基本的なチュートリアルの実行
```bash
python glmm_tutorial.py
```

### 実践例の実行
```bash
python glmm_practical_examples.py
```

## 🎯 学習の進め方

### 初心者向け
1. **`glmm_tutorial.py`** から始める
   - 基本概念の理解
   - サンプルデータの可視化
   - 理論的背景の学習

2. **`glmm_theory.md`** で理論を深める
   - 数学的定式化の理解
   - 推定方法の学習

### 中級者向け
1. **`glmm_practical_examples.py`** で実装を学ぶ
   - 実際のコード例の理解
   - 様々なパッケージの使用方法

2. 自分のデータでの実践
   - 適切なモデルの選択
   - パラメータの推定
   - 結果の解釈

### 上級者向け
1. ベイズGLMMの実装
2. 高次元データへの応用
3. 非線形効果のモデリング

## 📊 GLMMの基本概念

### 固定効果（Fixed Effects）
- 全体的な傾向を表すパラメータ
- 全てのグループに共通する効果
- 例：年齢、性別、教育レベルなど

### ランダム効果（Random Effects）
- グループ間の変動を表すパラメータ
- 各グループ固有の効果
- 例：学校、病院、地域など

### リンク関数（Link Function）
- 応答変数を線形予測子に変換する関数
- 例：ロジット関数、プロビット関数、対数関数など

### 分布族（Distribution Family）
- 応答変数の確率分布
- 例：二項分布、ポアソン分布、正規分布など

## 🔧 実装に使用するパッケージ

### Python
- **statsmodels**: 線形混合効果モデル
- **PyMC**: ベイズGLMM
- **scikit-learn**: 基本的な機械学習手法

### R（オプション）
- **lme4**: 線形・一般化線形混合効果モデル
- **nlme**: 非線形混合効果モデル
- **glmmTMB**: 一般化線形混合効果モデル

## 📈 応用例

### 医学研究
- 患者の治療効果の分析
- 病院間の治療成績の比較
- 反復測定データの分析

### 教育研究
- 生徒の学力の分析
- 学校間の学習効果の比較
- 縦断データの分析

### 社会科学
- 地域間の政策効果の比較
- 世帯調査データの分析
- パネルデータの分析

### 生態学
- 個体群の生存率の分析
- 種間の相互作用の分析
- 時系列データの分析

## ⚠️ 注意点とベストプラクティス

### モデル選択
- 情報量基準（AIC, BIC）の使用
- 交差検証の実施
- 残差分析によるモデル診断

### 収束性の確認
- 対数尤度の収束
- パラメータ推定値の安定性
- 標準誤差の妥当性

### 多重比較
- 多重比較補正の適用
- 偽発見率の制御
- 事前計画された比較の活用

## 📚 参考文献

1. McCulloch, C. E., & Searle, S. R. (2001). Generalized, Linear, and Mixed Models. Wiley.
2. Pinheiro, J. C., & Bates, D. M. (2000). Mixed-Effects Models in S and S-PLUS. Springer.
3. Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press.
4. Zuur, A. F., Ieno, E. N., Walker, N. J., Saveliev, A. A., & Smith, G. M. (2009). Mixed Effects Models and Extensions in Ecology with R. Springer.
5. Stroup, W. W. (2012). Generalized Linear Mixed Models: Modern Concepts, Methods and Applications. CRC Press.

## 🤝 貢献

このリソースの改善提案や質問がある場合は、Issueを作成してください。

## 📄 ライセンス

このプロジェクトは教育目的で作成されており、自由に使用・改変できます。

---

**GLMMの学習を楽しんでください！** 🎉

何か質問があれば、お気軽にお聞きください。
# 混合ハザードモデル（Mixture Hazard Models）学習プロジェクト

このリポジトリは混合ハザードモデルについて体系的に学習するためのコンテンツを提供します。

## 学習目標
- 混合ハザードモデルの基本概念と理論的背景の理解
- ジョイント混合モデル（Joint Mixture Models）の応用
- 実データを用いた実装とモデリング技術の習得
- 生存時間解析における最新手法の理解

## 学習コンテンツ
1. 理論編：基本概念から応用まで
2. 実装編：PythonとRによる実践
3. 事例研究：医学・工学分野での応用例
4. 練習問題：段階的な理解確認

学習を開始するには `docs/01_introduction.md` から始めてください。
