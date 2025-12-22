# 放物運動シミュレーター / Projectile Motion Simulator

空気抵抗がある場合とない場合のボールの放物運動を可視化・比較できる物理学習支援アプリケーションです。

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 概要 / Overview

このシミュレーターでは、以下のことが学べます：

- **空気抵抗なしの理想的な放物運動**（解析解）
- **空気抵抗ありの現実的な放物運動**（Runge-Kutta法による数値解法）
- 両者の軌跡を同時に表示して比較
- パラメーターをリアルタイムで変更して即座に結果を確認

---

## 必要環境 / Requirements

```bash
Python 3.7+
numpy
matplotlib
tkinter (通常Pythonに標準で付属)
```

### インストール / Installation

```bash
pip install numpy matplotlib
```

---

## 使い方 / How to Use

### 起動方法

```bash
python projectile_motion_simulator.py
```

### 画面構成

```
┌─────────────────────────────────────────────────────────────┐
│  Control Panel / コントロールパネル   │  Trajectory Plot    │
│  ┌──────────────────────────────┐   │  (軌跡プロット)      │
│  │ Launch Parameters            │   │                      │
│  │  • Initial Velocity 初速度   │   │     ⌒⌒⌒ (青線)     │
│  │  • Angle 発射角度            │   │   ⌒     ⌒         │
│  │  • Initial Height 初期高さ   │   │ ⌒   ⌒⌒ (赤線)    │
│  ├──────────────────────────────┤   │ 🟢────────────────   │
│  │ Ball Properties              │   │                      │
│  │  • Mass 質量                 │   │ 青: 空気抵抗なし     │
│  │  • Radius 半径               │   │ 赤: 空気抵抗あり     │
│  │  • Drag Coefficient 抗力係数 │   │                      │
│  ├──────────────────────────────┤   │                      │
│  │ Display Options              │   │                      │
│  │  ☑ No Air Resistance        │   │                      │
│  │  ☑ With Air Resistance      │   │                      │
│  ├──────────────────────────────┤   │                      │
│  │ 🚀 Launch  🔄 Reset  ❓ Help │   │                      │
│  ├──────────────────────────────┤   │                      │
│  │ Results / 結果               │   │                      │
│  │  飛行時間、最大高さ、飛距離   │   │                      │
│  └──────────────────────────────┘   │                      │
└─────────────────────────────────────────────────────────────┘
```

---

## パラメーター説明 / Parameter Description

### 発射パラメーター (Launch Parameters)

| パラメーター | 英語 | 説明 | 単位 | 範囲 |
|-------------|------|------|------|------|
| 初速度 | Initial Velocity (v₀) | ボールを投げる速さ | m/s | 1-100 |
| 発射角度 | Angle (θ) | 水平面からの角度 | 度 (°) | 0-90 |
| 初期高さ | Initial Height (h₀) | 発射点の地面からの高さ | m | 0-50 |

### ボールの性質 (Ball Properties)

| パラメーター | 英語 | 説明 | 単位 | 範囲 |
|-------------|------|------|------|------|
| 質量 | Mass (m) | ボールの重さ | kg | 0.01-1.0 |
| 半径 | Radius (r) | ボールの半径 | m | 0.01-0.2 |
| 抗力係数 | Drag Coefficient (Cd) | 空気抵抗の強さを決める係数 | - | 0.1-1.5 |

---

## プリセットボール / Preset Balls

よく使われるボールのプリセットを用意しています：

| ボール | 質量 | 半径 | 抗力係数 |
|--------|------|------|----------|
| ⚾ 野球 (Baseball) | 0.145 kg | 0.0366 m | 0.47 |
| ⚽ サッカー (Soccer) | 0.43 kg | 0.11 m | 0.25 |
| ⛳ ゴルフ (Golf) | 0.046 kg | 0.0214 m | 0.25 |

---

## 物理学の背景 / Physics Background

### 空気抵抗なし (Ideal Projectile Motion)

解析解で計算可能な理想的な放物運動：

```
運動方程式:
  x(t) = v₀ · cos(θ) · t
  y(t) = h₀ + v₀ · sin(θ) · t - ½ · g · t²

最大高さ:
  H = h₀ + (v₀ · sin(θ))² / (2g)

飛距離 (h₀ = 0 の場合):
  R = v₀² · sin(2θ) / g

最大飛距離の角度: θ = 45°
```

### 空気抵抗あり (With Air Resistance)

二次抵抗モデルを使用した現実的なシミュレーション：

```
空気抵抗力:
  F_drag = ½ · ρ · Cd · A · v²

  ρ  = 空気密度 (1.225 kg/m³ at 海面, 15°C)
  Cd = 抗力係数 (球体 ≈ 0.47)
  A  = 断面積 (π · r²)
  v  = 速度の大きさ

運動方程式 (数値解法):
  dvₓ/dt = -k · v · vₓ
  dvᵧ/dt = -g - k · v · vᵧ

  k = ½ · ρ · Cd · A / m

数値解法: 4次 Runge-Kutta法
```

---

## 学習ポイント / Learning Points

### 1. 空気抵抗の影響を観察する

**実験**: 同じ初速度・角度で、空気抵抗の有無を比較

**観察ポイント**:
- 空気抵抗ありでは飛距離が減少する
- 軌跡が非対称になる（上昇より下降が急）
- 最高点の位置が手前にずれる

### 2. 最適発射角度を探す

**実験**: 角度を0°から90°まで変化させて飛距離を観察

**発見**:
- 空気抵抗なし: 45°で最大飛距離
- 空気抵抗あり: 最適角度は45°より小さい（30-40°程度）

### 3. ボールの質量と大きさの影響

**実験**: 同じ速度・角度で、質量や半径を変える

**発見**:
- 重いボール → 空気抵抗の影響が小さい
- 大きいボール → 断面積が大きく、抵抗が大きい
- 密度(質量/体積)が高いほど遠くへ飛ぶ

### 4. 速度による影響

**実験**: 初速度を変化させる

**発見**:
- 低速: 空気抵抗の影響は小さい
- 高速: 抵抗力 ∝ v² なので急激に影響が大きくなる

---

## 教育での活用例 / Educational Applications

### 中学校・高校物理

1. **等加速度運動の導入**
   - 空気抵抗なしモードで放物運動の基本を学ぶ
   - 初速度と発射角度の関係を実験

2. **力学の発展**
   - 空気抵抗を加えて「なぜ野球の球は曲がるか」を考える
   - 実際のスポーツと物理の関係を理解

### 大学物理・工学

1. **数値計算入門**
   - Runge-Kutta法の実例として
   - 解析解と数値解の比較

2. **流体力学の導入**
   - 抗力係数と形状の関係
   - レイノルズ数の概念へ

---

## トラブルシューティング / Troubleshooting

### Q: 起動時にエラーが出る

```bash
# tkinterがない場合
sudo apt-get install python3-tk  # Ubuntu/Debian
brew install python-tk           # macOS
```

### Q: 日本語が文字化けする

matplotlib の日本語フォント設定が必要な場合があります：

```python
# ~/.config/matplotlib/matplotlibrc に追加
font.family: sans-serif
font.sans-serif: DejaVu Sans, Noto Sans CJK JP, ...
```

### Q: 描画が遅い

- 時間刻み (dt) が小さすぎる可能性があります
- 初速度を下げてテストしてください

---

## ライセンス / License

MIT License - 自由にご使用ください。

---

## 物理定数 / Physical Constants Used

| 定数 | 値 | 説明 |
|------|-----|------|
| 重力加速度 g | 9.81 m/s² | 地球表面での標準値 |
| 空気密度 ρ | 1.225 kg/m³ | 海面、15°C での値 |

---

## 参考文献 / References

1. 「力学」(原島鮮) - 日本の定番力学教科書
2. "Classical Mechanics" (Goldstein) - 古典力学の標準テキスト
3. "The Physics of Baseball" (Robert K. Adair) - 野球の物理学

---

*Created for physics education and learning support*
