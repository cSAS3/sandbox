#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放物運動シミュレーター
======================
空気抵抗がある場合とない場合のボールの放物運動を
シミュレーションし、学習を支援するGUIアプリケーション

物理モデル:
- 空気抵抗なし: 古典的な放物運動（解析解）
- 空気抵抗あり: 二次抵抗モデル F_drag = -0.5 * ρ * Cd * A * v² （数値解法）
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('TkAgg')

# 日本語フォント設定（利用可能なフォントを試行）
try:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
except:
    pass


class PhysicsEngine:
    """物理演算エンジン"""

    # 物理定数
    GRAVITY = 9.81  # 重力加速度 [m/s²]
    AIR_DENSITY = 1.225  # 空気密度 [kg/m³]（海面、15℃）

    @staticmethod
    def calculate_no_air_resistance(v0: float, angle_deg: float,
                                     height: float = 0, dt: float = 0.01):
        """
        空気抵抗なしの放物運動（解析解）

        Parameters:
        -----------
        v0 : float - 初速度 [m/s]
        angle_deg : float - 発射角度 [度]
        height : float - 初期高さ [m]
        dt : float - 時間刻み [s]

        Returns:
        --------
        tuple: (x座標配列, y座標配列, 飛行時間, 最大高さ, 飛距離)
        """
        angle_rad = np.radians(angle_deg)
        v0x = v0 * np.cos(angle_rad)
        v0y = v0 * np.sin(angle_rad)
        g = PhysicsEngine.GRAVITY

        # 飛行時間の計算（y = 0 になる時刻）
        # y = height + v0y*t - 0.5*g*t² = 0
        # 二次方程式の解
        discriminant = v0y**2 + 2 * g * height
        if discriminant < 0:
            return np.array([0]), np.array([height]), 0, height, 0

        t_flight = (v0y + np.sqrt(discriminant)) / g

        # 時間配列
        t = np.arange(0, t_flight + dt, dt)

        # 軌跡の計算
        x = v0x * t
        y = height + v0y * t - 0.5 * g * t**2

        # y < 0 の部分を除去
        valid_indices = y >= 0
        x = x[valid_indices]
        y = y[valid_indices]

        # 最大高さ
        max_height = height + v0y**2 / (2 * g) if v0y > 0 else height

        # 飛距離
        range_distance = x[-1] if len(x) > 0 else 0

        return x, y, t_flight, max_height, range_distance

    @staticmethod
    def calculate_with_air_resistance(v0: float, angle_deg: float,
                                       mass: float, radius: float,
                                       drag_coefficient: float = 0.47,
                                       height: float = 0, dt: float = 0.001):
        """
        空気抵抗ありの放物運動（Runge-Kutta法による数値解法）

        空気抵抗モデル: F_drag = 0.5 * ρ * Cd * A * v²

        Parameters:
        -----------
        v0 : float - 初速度 [m/s]
        angle_deg : float - 発射角度 [度]
        mass : float - ボールの質量 [kg]
        radius : float - ボールの半径 [m]
        drag_coefficient : float - 抗力係数 Cd（球体は約0.47）
        height : float - 初期高さ [m]
        dt : float - 時間刻み [s]

        Returns:
        --------
        tuple: (x座標配列, y座標配列, 飛行時間, 最大高さ, 飛距離)
        """
        angle_rad = np.radians(angle_deg)
        g = PhysicsEngine.GRAVITY
        rho = PhysicsEngine.AIR_DENSITY

        # 断面積
        A = np.pi * radius**2

        # 抵抗定数 k = 0.5 * ρ * Cd * A / m
        k = 0.5 * rho * drag_coefficient * A / mass

        # 初期条件
        x, y = 0.0, height
        vx = v0 * np.cos(angle_rad)
        vy = v0 * np.sin(angle_rad)

        # 結果を格納するリスト
        x_list = [x]
        y_list = [y]
        t = 0
        max_height = height

        # 4次Runge-Kutta法
        while y >= 0 and t < 1000:  # 安全のため最大1000秒
            # 速度の大きさ
            v = np.sqrt(vx**2 + vy**2)

            # 加速度の計算
            if v > 0:
                ax = -k * v * vx
                ay = -g - k * v * vy
            else:
                ax = 0
                ay = -g

            # Runge-Kutta 4次 (簡易版)
            # k1
            k1_vx = ax * dt
            k1_vy = ay * dt
            k1_x = vx * dt
            k1_y = vy * dt

            # k2
            vx_mid = vx + k1_vx / 2
            vy_mid = vy + k1_vy / 2
            v_mid = np.sqrt(vx_mid**2 + vy_mid**2)
            ax_mid = -k * v_mid * vx_mid if v_mid > 0 else 0
            ay_mid = -g - k * v_mid * vy_mid if v_mid > 0 else -g
            k2_vx = ax_mid * dt
            k2_vy = ay_mid * dt
            k2_x = vx_mid * dt
            k2_y = vy_mid * dt

            # k3
            vx_mid = vx + k2_vx / 2
            vy_mid = vy + k2_vy / 2
            v_mid = np.sqrt(vx_mid**2 + vy_mid**2)
            ax_mid = -k * v_mid * vx_mid if v_mid > 0 else 0
            ay_mid = -g - k * v_mid * vy_mid if v_mid > 0 else -g
            k3_vx = ax_mid * dt
            k3_vy = ay_mid * dt
            k3_x = vx_mid * dt
            k3_y = vy_mid * dt

            # k4
            vx_end = vx + k3_vx
            vy_end = vy + k3_vy
            v_end = np.sqrt(vx_end**2 + vy_end**2)
            ax_end = -k * v_end * vx_end if v_end > 0 else 0
            ay_end = -g - k * v_end * vy_end if v_end > 0 else -g
            k4_vx = ax_end * dt
            k4_vy = ay_end * dt
            k4_x = vx_end * dt
            k4_y = vy_end * dt

            # 更新
            x += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
            y += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
            vx += (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6
            vy += (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6
            t += dt

            if y > max_height:
                max_height = y

            if y >= 0:
                x_list.append(x)
                y_list.append(y)

        x_array = np.array(x_list)
        y_array = np.array(y_list)
        range_distance = x_array[-1] if len(x_array) > 0 else 0

        return x_array, y_array, t, max_height, range_distance


class ProjectileMotionSimulator:
    """放物運動シミュレーターGUIアプリケーション"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("放物運動シミュレーター - Projectile Motion Simulator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # 変数の初期化
        self.init_variables()

        # GUIコンポーネントの作成
        self.create_widgets()

        # 初期プロットの表示
        self.update_plot()

    def init_variables(self):
        """変数の初期化"""
        self.v0 = tk.DoubleVar(value=30.0)          # 初速度 [m/s]
        self.angle = tk.DoubleVar(value=45.0)       # 発射角度 [度]
        self.height = tk.DoubleVar(value=0.0)       # 初期高さ [m]
        self.mass = tk.DoubleVar(value=0.145)       # 質量 [kg] (野球ボール)
        self.radius = tk.DoubleVar(value=0.0366)    # 半径 [m] (野球ボール)
        self.drag_coef = tk.DoubleVar(value=0.47)   # 抗力係数 (球体)

        self.show_no_air = tk.BooleanVar(value=True)    # 空気抵抗なし表示
        self.show_with_air = tk.BooleanVar(value=True)  # 空気抵抗あり表示
        self.show_velocity = tk.BooleanVar(value=False) # 速度ベクトル表示

    def create_widgets(self):
        """GUIウィジェットの作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左側：コントロールパネル
        self.create_control_panel(main_frame)

        # 右側：プロットエリア
        self.create_plot_area(main_frame)

    def create_control_panel(self, parent):
        """コントロールパネルの作成"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel / コントロールパネル",
                                        padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # === 発射パラメーター ===
        param_frame = ttk.LabelFrame(control_frame, text="Launch Parameters / 発射パラメーター",
                                      padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        # 初速度
        self.create_slider(param_frame, "Initial Velocity / 初速度 [m/s]:",
                          self.v0, 1, 100, 0)

        # 発射角度
        self.create_slider(param_frame, "Angle / 発射角度 [°]:",
                          self.angle, 0, 90, 1)

        # 初期高さ
        self.create_slider(param_frame, "Initial Height / 初期高さ [m]:",
                          self.height, 0, 50, 2)

        # === ボールのプロパティ ===
        ball_frame = ttk.LabelFrame(control_frame, text="Ball Properties / ボールの性質",
                                     padding=10)
        ball_frame.pack(fill=tk.X, pady=(0, 10))

        # 質量
        self.create_slider(ball_frame, "Mass / 質量 [kg]:",
                          self.mass, 0.01, 1.0, 0, resolution=0.01)

        # 半径
        self.create_slider(ball_frame, "Radius / 半径 [m]:",
                          self.radius, 0.01, 0.2, 1, resolution=0.001)

        # 抗力係数
        self.create_slider(ball_frame, "Drag Coefficient / 抗力係数 Cd:",
                          self.drag_coef, 0.1, 1.5, 2, resolution=0.01)

        # プリセットボタン
        preset_frame = ttk.Frame(ball_frame)
        preset_frame.grid(row=3, column=0, columnspan=2, pady=5)

        ttk.Button(preset_frame, text="Baseball/野球",
                   command=lambda: self.set_preset(0.145, 0.0366, 0.47)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Soccer/サッカー",
                   command=lambda: self.set_preset(0.43, 0.11, 0.25)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Golf/ゴルフ",
                   command=lambda: self.set_preset(0.046, 0.0214, 0.25)).pack(side=tk.LEFT, padx=2)

        # === 表示オプション ===
        display_frame = ttk.LabelFrame(control_frame, text="Display Options / 表示オプション",
                                        padding=10)
        display_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(display_frame, text="No Air Resistance / 空気抵抗なし",
                       variable=self.show_no_air,
                       command=self.update_plot).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="With Air Resistance / 空気抵抗あり",
                       variable=self.show_with_air,
                       command=self.update_plot).pack(anchor=tk.W)

        # === アクションボタン ===
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="🚀 Launch / 発射",
                   command=self.update_plot,
                   style='Accent.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🔄 Reset / リセット",
                   command=self.reset_values).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="❓ Help / ヘルプ",
                   command=self.show_help).pack(fill=tk.X, pady=2)

        # === 結果表示 ===
        self.result_frame = ttk.LabelFrame(control_frame,
                                           text="Results / 結果", padding=10)
        self.result_frame.pack(fill=tk.X, pady=(0, 10))

        self.result_text = tk.Text(self.result_frame, height=12, width=35,
                                   font=('Courier', 9), state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH)

        # === 物理式の表示 ===
        formula_frame = ttk.LabelFrame(control_frame,
                                        text="Physics Formulas / 物理式", padding=10)
        formula_frame.pack(fill=tk.X)

        formula_text = """Without Air Resistance:
  x(t) = v₀·cos(θ)·t
  y(t) = h₀ + v₀·sin(θ)·t - ½gt²

With Air Resistance:
  F_drag = ½·ρ·Cd·A·v²
  a = F/m = -g - (k·v)·v̂
  (Numerical: Runge-Kutta 4th)"""

        ttk.Label(formula_frame, text=formula_text,
                  font=('Courier', 8)).pack()

    def create_slider(self, parent, label, variable, from_, to_, row, resolution=1):
        """スライダーの作成"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)

        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=row, column=1, sticky=tk.EW, pady=2)

        slider = ttk.Scale(slider_frame, from_=from_, to=to_, variable=variable,
                          orient=tk.HORIZONTAL, length=150,
                          command=lambda x: self.on_slider_change())
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        value_label = ttk.Label(slider_frame, textvariable=variable, width=8)
        value_label.pack(side=tk.RIGHT)

        # 値の表示を更新
        def update_label(*args):
            if resolution < 1:
                value_label.config(text=f"{variable.get():.3f}")
            else:
                value_label.config(text=f"{variable.get():.1f}")

        variable.trace_add('write', update_label)
        update_label()

    def on_slider_change(self):
        """スライダー変更時のコールバック（リアルタイム更新）"""
        self.update_plot()

    def create_plot_area(self, parent):
        """プロットエリアの作成"""
        plot_frame = ttk.LabelFrame(parent, text="Trajectory Plot / 軌跡プロット",
                                     padding=5)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Matplotlib Figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        """プロットの更新"""
        # パラメーターの取得
        v0 = self.v0.get()
        angle = self.angle.get()
        height = self.height.get()
        mass = self.mass.get()
        radius = self.radius.get()
        drag_coef = self.drag_coef.get()

        # プロットのクリア
        self.ax.clear()

        results = []

        # 空気抵抗なしの軌跡
        if self.show_no_air.get():
            x_no_air, y_no_air, t1, h1, r1 = PhysicsEngine.calculate_no_air_resistance(
                v0, angle, height
            )
            self.ax.plot(x_no_air, y_no_air, 'b-', linewidth=2,
                        label='No Air Resistance / 空気抵抗なし')
            results.append(("No Air Resistance", t1, h1, r1))

        # 空気抵抗ありの軌跡
        if self.show_with_air.get():
            x_with_air, y_with_air, t2, h2, r2 = PhysicsEngine.calculate_with_air_resistance(
                v0, angle, mass, radius, drag_coef, height
            )
            self.ax.plot(x_with_air, y_with_air, 'r-', linewidth=2,
                        label='With Air Resistance / 空気抵抗あり')
            results.append(("With Air Resistance", t2, h2, r2))

        # 発射点のマーカー
        self.ax.plot(0, height, 'go', markersize=10, label='Launch Point / 発射点')

        # 初速度ベクトルの描画
        arrow_scale = 0.5
        dx = v0 * np.cos(np.radians(angle)) * arrow_scale
        dy = v0 * np.sin(np.radians(angle)) * arrow_scale
        self.ax.annotate('', xy=(dx, height + dy), xytext=(0, height),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))

        # グラフの設定
        self.ax.set_xlabel('Distance / 距離 [m]', fontsize=12)
        self.ax.set_ylabel('Height / 高さ [m]', fontsize=12)
        self.ax.set_title(f'Projectile Motion / 放物運動\n'
                         f'v₀={v0:.1f} m/s, θ={angle:.1f}°, h₀={height:.1f} m',
                         fontsize=14)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(bottom=0)
        self.ax.set_aspect('equal', adjustable='datalim')

        # 地面のライン
        xlim = self.ax.get_xlim()
        self.ax.axhline(y=0, color='brown', linewidth=2, linestyle='-')
        self.ax.fill_between([xlim[0], xlim[1]*1.1], -1, 0, color='green', alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

        # 結果の表示
        self.display_results(results, v0, angle, height, mass, radius, drag_coef)

    def display_results(self, results, v0, angle, height, mass, radius, drag_coef):
        """結果の表示"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        text = f"""=== Input Parameters ===
Initial Velocity: {v0:.2f} m/s
Launch Angle: {angle:.1f}°
Initial Height: {height:.2f} m
Ball Mass: {mass:.3f} kg
Ball Radius: {radius:.4f} m
Drag Coefficient: {drag_coef:.2f}

=== Results ===
"""

        for name, t, h, r in results:
            if "No Air" in name:
                text += f"\n[No Air Resistance / 空気抵抗なし]\n"
            else:
                text += f"\n[With Air Resistance / 空気抵抗あり]\n"
            text += f"  Flight Time: {t:.2f} s\n"
            text += f"  Max Height: {h:.2f} m\n"
            text += f"  Range: {r:.2f} m\n"

        # 差分の計算
        if len(results) == 2:
            t_diff = results[0][1] - results[1][1]
            h_diff = results[0][2] - results[1][2]
            r_diff = results[0][3] - results[1][3]
            r_pct = (r_diff / results[0][3] * 100) if results[0][3] > 0 else 0

            text += f"\n=== Difference / 差 ===\n"
            text += f"Range reduction: {r_diff:.2f} m ({r_pct:.1f}%)\n"
            text += f"Max height diff: {h_diff:.2f} m\n"

        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

    def set_preset(self, mass, radius, drag):
        """プリセット値の設定"""
        self.mass.set(mass)
        self.radius.set(radius)
        self.drag_coef.set(drag)
        self.update_plot()

    def reset_values(self):
        """値のリセット"""
        self.v0.set(30.0)
        self.angle.set(45.0)
        self.height.set(0.0)
        self.mass.set(0.145)
        self.radius.set(0.0366)
        self.drag_coef.set(0.47)
        self.show_no_air.set(True)
        self.show_with_air.set(True)
        self.update_plot()

    def show_help(self):
        """ヘルプダイアログの表示"""
        help_text = """
【放物運動シミュレーター - 使い方】

■ 基本操作
1. スライダーで各パラメーターを調整します
2. 軌跡はリアルタイムで更新されます
3. 青線=空気抵抗なし、赤線=空気抵抗あり

■ パラメーター説明
• 初速度 (v₀): ボールを投げる速さ [m/s]
• 発射角度 (θ): 水平からの角度 [度]
• 初期高さ (h₀): 発射位置の高さ [m]
• 質量 (m): ボールの重さ [kg]
• 半径 (r): ボールの大きさ [m]
• 抗力係数 (Cd): 空気抵抗の強さ（球=0.47）

■ プリセットボール
• 野球: 145g, 直径7.3cm, Cd=0.47
• サッカー: 430g, 直径22cm, Cd=0.25
• ゴルフ: 46g, 直径4.3cm, Cd=0.25

■ 物理学のポイント
• 空気抵抗なし：45°が最大飛距離
• 空気抵抗あり：最適角度は45°より小さくなる
• 重いボールは空気抵抗の影響が小さい
• 速度が速いほど空気抵抗の影響が大きい

■ 学習のヒント
1. 空気抵抗の有無で軌跡を比較
2. 角度を変えて最大飛距離を探す
3. ボールの大きさ・重さの影響を観察
"""
        messagebox.showinfo("Help / ヘルプ", help_text)

    def run(self):
        """アプリケーションの実行"""
        self.root.mainloop()


def main():
    """メイン関数"""
    print("=" * 50)
    print("放物運動シミュレーター")
    print("Projectile Motion Simulator")
    print("=" * 50)
    print("Starting GUI application...")
    print("Close the window to exit.\n")

    app = ProjectileMotionSimulator()
    app.run()


if __name__ == "__main__":
    main()
