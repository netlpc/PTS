'''- Improved result display: numerical outputs are rounded to 3 decimal places by default.
- Preserved exact symbolic results internally and added a "View Exact Result" button.
- Separated computation precision from UI presentation to improve readability.
- Introduced a unified result formatting utility for consistent output handling.
'''
import tkinter as tk
import time, re, threading
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp
from sympy import lambdify, symbols, integrate, nsimplify, pretty
from scipy.integrate import quad, simpson
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from queue import Queue
# --- SymPy parser imports for implicit multiplication
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)


SAFE_LOCALS = {
    "pi": sp.pi, "E": sp.E, "e": sp.E, "I": sp.I,
    "inf": sp.oo, "-inf": -sp.oo,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log, "exp": sp.exp,
    "abs": sp.Abs, "sign": sp.sign,
}



x_sym = symbols('x')

# ===== Result formatting & exact-value storage =====
last_raw_result = None

def format_result_for_display(expr, max_decimals=3):
    """
    Returns:
        display_str: formatted string for UI
        raw_str: exact, unmodified SymPy string
    """
    raw_str = sp.sstr(expr)
    try:
        val = float(expr.evalf())
        display_str = f"{val:.{max_decimals}f}"
        return display_str, raw_str
    except Exception:
        return raw_str, raw_str

def format_function_input(func_str: str) -> str:
    return re.sub(r'(?<=\d)([a-zA-Z])', r'*\1', func_str)

# --- Use SymPy's implicit multiplication parser
def parse_input_to_sympy(value: str):
    value = value.strip()
    if value == "":
        raise ValueError("Empty input.")
    try:
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(value, local_dict=SAFE_LOCALS, transformations=transformations)
        return expr
    except Exception as e:
        raise ValueError(f"Invalid input: {e}")

def parse_input_to_float(value: str) -> float:
    expr = parse_input_to_sympy(value)
    if expr == sp.oo:
        return np.inf
    if expr == -sp.oo:
        return -np.inf
    return float(expr.evalf())

def evaluate_symbolic_function(func_str: str) -> sp.Expr:
    try:
        s = func_str.replace('^', '**').replace('ln', 'log')
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(s, local_dict=SAFE_LOCALS, transformations=transformations)
        if len(expr.free_symbols - {x_sym}) > 0:
            raise ValueError("Only variable x is allowed.")
        return expr
    except Exception as e:
        raise ValueError(f"Error parsing symbolic function: {e}")

def build_numeric_callable(expr: sp.Expr):
    try:
        f = lambdify(x_sym, expr, modules="numpy")
        _ = f(0.0)
        return f
    except Exception as e:
        raise ValueError(f"Failed to build numeric function: {e}")

def ensure_finite_on_probe(func, a, b, n_probe=9):
    if not (np.isfinite(a) and np.isfinite(b)):
        return True, None
    xs = np.linspace(a, b, n_probe)
    try:
        vals = np.asarray(func(xs), dtype=float).reshape(-1)
    except Exception:
        return False, None
    if not np.all(np.isfinite(vals)):
        bad_idx = np.where(~np.isfinite(vals))[0]
        xbad = xs[bad_idx[0]] if bad_idx.size > 0 else None
        return False, xbad
    return True, None

def evaluate_function(x_vals, func_str: str):
    expr = evaluate_symbolic_function(func_str)
    f = build_numeric_callable(expr)
    y = np.asarray(f(np.array(x_vals, dtype=float)), dtype=float).reshape(-1)
    if not np.all(np.isfinite(y)):
        raise ValueError("Function evaluation returned non-finite values.")
    return y

# =========================
# ====== Plot (Embedded) ==
# =========================

def init_plot_area(parent_frame):
    fig = plt.Figure(figsize=(8, 3.6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Function Graph")
    ax.grid(True)
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, parent_frame, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    return fig, ax, canvas, toolbar

def clear_plot(ax, canvas):
    ax.clear()
    ax.set_title("Function Graph")
    ax.grid(True)
    canvas.draw_idle()

def plot_embedded(func_str: str, lower: float, upper: float):
    try:
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            clear_plot(plot_ax, plot_canvas)
            return
        x_vals = np.linspace(lower, upper, 500)
        y_vals = evaluate_function(x_vals, func_str)
        plot_ax.clear()
        plot_ax.plot(x_vals, y_vals, label=f"f(x) = {func_str}")
        plot_ax.axhline(0, linewidth=0.8, linestyle='--')
        plot_ax.axvline(0, linewidth=0.8, linestyle='--')
        plot_ax.set_title(f"Function Graph: {func_str}")
        plot_ax.set_xlabel("x")
        plot_ax.set_ylabel("f(x)")
        plot_ax.grid(True)
        plot_ax.legend()
        plot_canvas.draw_idle()
    except Exception:
        clear_plot(plot_ax, plot_canvas)

# =========================
# ====== Numeric Rules ====
# =========================

def composite_simpson_38(func, a, b, n_intervals=300):
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("Simpson 3/8 requires finite limits.")
    if a == b:
        return 0.0
    n = int(max(3, n_intervals))
    if n % 3 != 0:
        n += (3 - n % 3)
    x = np.linspace(a, b, n + 1)
    y = np.asarray(func(x), dtype=float).reshape(-1)
    if not np.all(np.isfinite(y)):
        raise ValueError("Function produced non-finite values in Simpson 3/8.")
    h = (b - a) / n
    S = y[0] + y[-1]
    S += 3 * np.sum(y[1:-1][np.array([i % 3 != 0 for i in range(1, n)])])
    S += 2 * np.sum(y[3:-3:3])
    return 3 * h / 8 * S

def romberg_custom(func, a, b, max_level=8, tol=1e-8):
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("Romberg requires finite limits.")
    if a == b:
        return 0.0
    R = np.zeros((max_level+1, max_level+1), dtype=float)
    fa = float(np.asarray(func(a)).reshape(-1)[0])
    fb = float(np.asarray(func(b)).reshape(-1)[0])
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ValueError("Function non-finite at endpoints for Romberg.")
    h = (b - a)
    R[0, 0] = 0.5 * h * (fa + fb)
    for k in range(1, max_level+1):
        n_intervals = 2**(k-1)
        h *= 0.5
        xs_mid = a + h * (np.arange(1, 2*n_intervals, 2))
        fm = np.asarray(func(xs_mid), dtype=float).reshape(-1)
        if not np.all(np.isfinite(fm)):
            raise ValueError("Function produced non-finite values in Romberg refinement.")
        R[k, 0] = 0.5 * R[k-1, 0] + h * np.sum(fm)
        for m in range(1, k+1):
            R[k, m] = R[k, m-1] + (R[k, m-1] - R[k-1, m-1]) / (4**m - 1)
        if abs(R[k, k] - R[k-1, k-1]) < tol * max(1.0, abs(R[k, k])):
            return R[k, k]
    return R[max_level, max_level]

def monte_carlo_stratified(func, a, b, n_samples=5000, n_strata=50, rng=None):
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("Monte Carlo requires finite limits.")
    if a == b:
        return 0.0
    rng = np.random.default_rng() if rng is None else rng
    n_strata = max(1, int(n_strata))
    n_samples = max(n_strata, int(n_samples))
    per = n_samples // n_strata
    if per == 0:
        per = 1
        n_strata = n_samples
    width = (b - a) / n_strata
    est = 0.0
    for s in range(n_strata):
        left = a + s * width
        right = left + width
        u = rng.random(per)
        xs = left + u * width
        vals = np.asarray(func(xs), dtype=float).reshape(-1)
        if not np.all(np.isfinite(vals)):
            raise ValueError("Function produced non-finite values in Monte Carlo.")
        est += width * np.mean(vals)
    return est

def gaussian_quadrature_fixed(f, a, b, n=64):
    """
    Fixed-order Gauss–Legendre quadrature.
    Replacement for deprecated scipy.integrate.quadrature.
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("Gaussian Quadrature requires finite limits.")
    if a == b:
        return 0.0
    xs, ws = np.polynomial.legendre.leggauss(n)  # on [-1,1]
    ts = 0.5*(b - a)*xs + 0.5*(a + b)
    vals = np.asarray(f(ts), dtype=float).reshape(-1)
    if not np.all(np.isfinite(vals)):
        raise ValueError("Function produced non-finite values in Gaussian Quadrature.")
    return 0.5*(b - a)*np.dot(ws, vals)

# =========================
# ====== Histories   ======
# =========================

history = []

def update_history(new_record):
    global history
    history.append(new_record)
    # No cap on history length; listbox already scrolls
    history_listbox.delete(0, tk.END)
    for record in history:
        history_listbox.insert(tk.END, record)

# =========================
# ====== Instructions =====
# =========================

instructions = {
    "English": [
        "Common mathematical functions and constants:",
        "log10(x) - Common logarithm (base 10), e.g., log10(100)",
        "ln(x) / log(x) - Natural logarithm (base e), e.g., ln(2)",
        "pi - Pi, input: pi",
        "sin(x), cos(x), tan(x)",
        "sqrt(x) - Square root, e.g., sqrt(4)",
        "exp(x) or e^x - Exponential function, e.g., exp(1) = e^1",
        "Infinity:",
        "Positive infinity - inf",
        "Negative infinity - -inf",
        "",
    ],
    "中文（简体）": [
        "常见数学函数和常数的使用方法：",
        "log10(x) - 常用对数（以10为底），例如：log10(100)",
        "ln(x) / log(x) - 自然对数（以e为底），例如：ln(2)",
        "pi - 圆周率，输入：pi",
        "sin(x), cos(x), tan(x)",
        "sqrt(x) - 平方根，例如：sqrt(4)",
        "exp(x) 或 e^x - 指数函数，例如：exp(1) = e^1",
        "表示无穷大的方法：",
        "正无穷大 - inf",
        "负无穷大 - -inf",
        "",
    ],
    "國語（繁体）": [
        "常見數學函數與常數的使用方式：",
        "log10(x) - 常用對數（以 10 為底），例如：log10(100)",
        "ln(x) / log(x) - 自然對數（以 e 為底），例如：ln(2)",
        "pi - 圓周率，輸入：pi",
        "sin(x), cos(x), tan(x)",
        "sqrt(x) - 平方根，例如：sqrt(4)",
        "exp(x) 或 e^x - 指數函數，例如：exp(1) = e^1",
        "表示無窮大的方式：",
        "正無窮大 - inf",
        "負無窮大 - -inf",
        "",
    ],
    "日本語": [
        "一般的な数学関数と定数の使用方法：",
        "log10(x) - 常用対数（底10）、例：log10(100)",
        "ln(x) / log(x) - 自然対数（底e）、例：ln(2)",
        "pi - 円周率，入力：pi",
        "sin(x), cos(x), tan(x)",
        "sqrt(x) - 平方根、例：sqrt(4)",
        "exp(x) または e^x - 指数関数、例：exp(1) = e^1",
        "無限大を表す方法：",
        "正の無限大 - inf",
        "負の無限大 - -inf",
        "",
    ],
    "한국어": [
        "자주 쓰는 수학 함수와 상수:",
        "log10(x) - 상용로그(밑 10), 예: log10(100)",
        "ln(x) / log(x) - 자연로그(밑 e), 예: ln(2)",
        "pi - 원주율, 입력: pi",
        "sin(x), cos(x), tan(x)",
        "sqrt(x) - 제곱근, 예: sqrt(4)",
        "exp(x) 또는 e^x - 지수함수, 예: exp(1) = e^1",
        "무한대를 표시하는 방법:",
        "양의 무한대 - inf",
        "음의 무한대 - -inf",
        "",
    ],
    "Español": [
        "Funciones matemáticas y constantes comunes:",
        "log10(x) - Logaritmo común (base 10), ej.: log10(100)",
        "ln(x) / log(x) - Logaritmo natural (base e), ej.: ln(2)",
        "pi - Pi, entrada: pi",
        "sin(x), cos(x), tan(x)",
        "sqrt(x) - Raíz cuadrada, ej.: sqrt(4)",
        "exp(x) o e^x - Función exponencial, ej.: exp(1) = e^1",
        "Representación de infinito:",
        "Infinito positivo - inf",
        "Infinito negativo - -inf",
        "",
    ],
    "Français": [
        "Fonctions mathématiques et constantes courantes :",
        "log10(x) - Logarithme décimal (base 10), ex. : log10(100)",
        "ln(x) / log(x) - Logarithme naturel (base e), ex. : ln(2)",
        "pi - Pi, entrée : pi",
        "sin(x), cos(x), tan(x)",
        "sqrt(x) - Racine carrée, ex. : sqrt(4)",
        "exp(x) ou e^x - Fonction exponentielle, ex. : exp(1) = e^1",
        "Représentation de l'infini :",
        "Infini positif - inf",
        "Infini négatif - -inf",
        "",
    ],
    "العربية": [
        "الدوال والثوابت الرياضية الشائعة:",
        "log10(x) - اللوغاريتم العشري (الأساس 10)، مثال: log10(100)",
        "ln(x) / log(x) - اللوغاريتم الطبيعي (الأساس e)، مثال: ln(2)",
        "pi - باي، الإدخال: pi",
        "sin(x), cos(x), tan(x)",
        "sqrt(x) - الجذر التربيعي، مثال: sqrt(4)",
        "exp(x) أو e^x - الدالة الأسية، مثال: exp(1) = e^1",
        "تمثيل اللانهاية:",
        "اللانهاية الموجبة - inf",
        "اللانهاية السالبة - -inf",
        "",
    ],
    "हिन्दी": [
        "सामान्य गणितीय फलन और नियतांक:",
        "log10(x) - सामान्य लघुगणक (आधार 10), उदाहरण: log10(100)",
        "ln(x) / log(x) - प्राकृतिक लघुगणक (आधार e), उदाहरण: ln(2)",
        "pi - पाई, इनपुट: pi",
        "sin(x), cos(x), tan(x)",
        "sqrt(x) - वर्गमूल, उदाहरण: sqrt(4)",
        "exp(x) या e^x - घातीय फलन, उदाहरण: exp(1) = e^1",
        "अनंत को दर्शाने के तरीके:",
        "धनात्मक अनंत - inf",
        "ऋणात्मक अनंत - -inf",
        "",
    ],
}

# =========================
# ========= GUI ===========
# =========================

root = tk.Tk()
root.title("Integration Calculator")

# Main vertical layout: controls (top) and plot (bottom)
main_paned = ttk.Panedwindow(root, orient=tk.VERTICAL)
main_paned.pack(fill=tk.BOTH, expand=True)

top_frame = ttk.Frame(main_paned)     # Notebook + Progress + History
bottom_frame = ttk.Frame(main_paned)  # Embedded Plot
main_paned.add(top_frame, weight=3)
main_paned.add(bottom_frame, weight=2)

# Bottom section: embedded plot
plot_fig, plot_ax, plot_canvas, plot_toolbar = init_plot_area(bottom_frame)

# Top section: language selector, instructions, notebook, progress bar, history
lang_var = tk.StringVar(value="English")
usage_window = None

lang_row = ttk.Frame(top_frame)
lang_row.pack(fill=tk.X, padx=10, pady=(8, 6))

lang_button = ttk.Combobox(
    lang_row,
    textvariable=lang_var,
    values=[
        "English",
        "中文（简体）",
        "國語（繁体）",
        "日本語",
        "한국어",
        "Español",
        "Français",
        "العربية",
        "हिन्दी"
    ],
    state="readonly",
    width=12
)
lang_button.pack(side=tk.LEFT)
usage_button = tk.Button(lang_row, text="Usage Instructions", bg="lightgreen")
usage_button.pack(side=tk.LEFT, padx=8)

# --- Middle area split: left (Notebook + Progress) | right (History) ---
mid_paned = ttk.Panedwindow(top_frame, orient=tk.HORIZONTAL)
mid_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=0)

left_panel = ttk.Frame(mid_paned)
right_panel = ttk.Frame(mid_paned, width=280)

mid_paned.add(left_panel, weight=3)
mid_paned.add(right_panel, weight=1)

# Notebook (left panel)
notebook = ttk.Notebook(left_panel)
notebook.pack(side=tk.TOP, expand=1, fill=tk.BOTH)

tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
tab3 = ttk.Frame(notebook)
notebook.add(tab1, text="Basic Integration")
notebook.add(tab2, text="Advanced Integration")
notebook.add(tab3, text="Improper Integral (Infinite)")

# Progress bar (below notebook)
progress = ttk.Progressbar(left_panel, orient=tk.HORIZONTAL, mode='determinate')
progress.pack(fill=tk.X, pady=6)

# History panel (right side, scrollable)
history_frame = ttk.Frame(right_panel)
history_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 10))

history_label = tk.Label(history_frame, text="History:", font=("Arial", 12, "bold"))
history_label.pack(anchor="w", padx=6, pady=(6, 2))

history_list_container = ttk.Frame(history_frame)
history_list_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

history_scrollbar = ttk.Scrollbar(history_list_container, orient=tk.VERTICAL)
history_listbox = tk.Listbox(history_list_container, height=14, yscrollcommand=history_scrollbar.set)
history_scrollbar.config(command=history_listbox.yview)

history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

def update_usage_instructions(lang):
    global usage_window
    # If no window or it has been closed, do nothing
    if usage_window is None or not usage_window.winfo_exists():
        return
    usage_text = "\n".join(instructions.get(lang, instructions["English"]))
    for widget in usage_window.winfo_children():
        widget.destroy()
    usage_label = tk.Label(usage_window, text=usage_text, justify="left", padx=10, pady=10)
    usage_label.pack()

def change_language(lang):
    if lang == "English":
        root.title("Integration Calculator")
        notebook.tab(0, text="Basic Integration")
        notebook.tab(1, text="Advanced Integration")
        notebook.tab(2, text="Improper Integral (Infinite)")
        usage_button.config(text="Usage Instructions")
        calc_button_tab1.config(text="Calculate Integral")
        reset_button_tab1.config(text="Reset")
        calc_button_tab2.config(text="Calculate Integral")
        reset_button_tab2.config(text="Reset")
        calc_button_tab3.config(text="Compute Integral")
        reset_button_tab3.config(text="Reset")
        method_label.config(text="Integration Method:")
        lower_label_tab2.config(text="Lower limit:")
        upper_label_tab2.config(text="Upper limit:")
        delta_label_tab2.config(text="Step size (for Numerical Integration):")
        lower_label_tab3.config(text="Lower limit:")
        upper_label_tab3.config(text="Upper limit:")
        func_label_tab3.config(text="Enter target function:")
        func_label_tab2.config(text="Enter target function:")
        numerical_method_label.config(text="Numerical Method:")
        result_label_tab1.config(text="")
        result_label_tab2.config(text="")
        result_label_tab3.config(text="")
        history_label.config(text="History:")
    elif lang == "中文（简体）":
        root.title("积分计算器")
        notebook.tab(0, text="基本积分")
        notebook.tab(1, text="进阶积分")
        notebook.tab(2, text="反常积分（无穷）")
        usage_button.config(text="使用说明")
        calc_button_tab1.config(text="计算积分")
        reset_button_tab1.config(text="重置")
        calc_button_tab2.config(text="计算积分")
        reset_button_tab2.config(text="重置")
        calc_button_tab3.config(text="计算积分")
        reset_button_tab3.config(text="重置")
        method_label.config(text="积分方法:")
        lower_label_tab2.config(text="下限:")
        upper_label_tab2.config(text="上限:")
        delta_label_tab2.config(text="步长（用于数值积分）:")
        lower_label_tab3.config(text="下限:")
        upper_label_tab3.config(text="上限:")
        func_label_tab3.config(text="输入目标函数:")
        func_label_tab2.config(text="输入目标函数:")
        numerical_method_label.config(text="数值方法:")
        result_label_tab1.config(text="")
        result_label_tab2.config(text="")
        result_label_tab3.config(text="")
        history_label.config(text="历史记录:")
    elif lang == "國語（繁体）":
        root.title("積分計算器")
        notebook.tab(0, text="基本積分")
        notebook.tab(1, text="進階積分")
        notebook.tab(2, text="反常積分（無窮）")
        usage_button.config(text="使用說明")
        calc_button_tab1.config(text="計算積分")
        reset_button_tab1.config(text="重置")
        calc_button_tab2.config(text="計算積分")
        reset_button_tab2.config(text="重置")
        calc_button_tab3.config(text="計算積分")
        reset_button_tab3.config(text="重置")
        method_label.config(text="積分方法:")
        lower_label_tab2.config(text="下限:")
        upper_label_tab2.config(text="上限:")
        delta_label_tab2.config(text="步長（數值積分）:")
        func_label_tab2.config(text="輸入目標函數:")
        func_label_tab3.config(text="輸入目標函數:")
        numerical_method_label.config(text="數值方法:")
        history_label.config(text="歷史紀錄:")
    elif lang == "日本語":
        root.title("積分計算機")
        notebook.tab(0, text="基本積分")
        notebook.tab(1, text="高度な積分")
        notebook.tab(2, text="異常積分（無限）")
        usage_button.config(text="使用説明")
        calc_button_tab1.config(text="積分を計算する")
        reset_button_tab1.config(text="リセット")
        calc_button_tab2.config(text="積分を計算する")
        reset_button_tab2.config(text="リセット")
        calc_button_tab3.config(text="積分を計算する")
        reset_button_tab3.config(text="リセット")
        method_label.config(text="積分方法:")
        lower_label_tab2.config(text="下限:")
        upper_label_tab2.config(text="上限:")
        delta_label_tab2.config(text="ステップサイズ（数値積分の場合）:")
        lower_label_tab3.config(text="下限:")
        upper_label_tab3.config(text="上限:")
        func_label_tab3.config(text="対象関数を入力:")
        func_label_tab2.config(text="対象関数を入力:")
        numerical_method_label.config(text="数値方法:")
        result_label_tab1.config(text="")
        result_label_tab2.config(text="")
        result_label_tab3.config(text="")
        history_label.config(text="履歴:")
    elif lang == "한국어":
        root.title("적분 계산기")
        notebook.tab(0, text="기본 적분")
        notebook.tab(1, text="고급 적분")
        notebook.tab(2, text="비정상 적분(무한)")
        usage_button.config(text="사용 안내")
        calc_button_tab1.config(text="적분 계산")
        reset_button_tab1.config(text="초기화")
        calc_button_tab2.config(text="적분 계산")
        reset_button_tab2.config(text="초기화")
        calc_button_tab3.config(text="적분 계산")
        reset_button_tab3.config(text="초기화")
        method_label.config(text="적분 방법:")
        lower_label_tab2.config(text="하한:")
        upper_label_tab2.config(text="상한:")
        delta_label_tab2.config(text="스텝 크기(수치 적분):")
        lower_label_tab3.config(text="하한:")
        upper_label_tab3.config(text="상한:")
        func_label_tab3.config(text="대상 함수 입력:")
        func_label_tab2.config(text="대상 함수 입력:")
        numerical_method_label.config(text="수치적 방법:")
        result_label_tab1.config(text="")
        result_label_tab2.config(text="")
        result_label_tab3.config(text="")
        history_label.config(text="기록:")
    elif lang == "Español":
        root.title("Calculadora de Integrales")
        notebook.tab(0, text="Integración básica")
        notebook.tab(1, text="Integración avanzada")
        notebook.tab(2, text="Integral impropia (infinita)")
        usage_button.config(text="Instrucciones de uso")
        calc_button_tab1.config(text="Calcular integral")
        reset_button_tab1.config(text="Restablecer")
        calc_button_tab2.config(text="Calcular integral")
        reset_button_tab2.config(text="Restablecer")
        calc_button_tab3.config(text="Calcular integral")
        reset_button_tab3.config(text="Restablecer")
        method_label.config(text="Método de integración:")
        lower_label_tab2.config(text="Límite inferior:")
        upper_label_tab2.config(text="Límite superior:")
        delta_label_tab2.config(text="Tamaño de paso (integración numérica):")
        lower_label_tab3.config(text="Límite inferior:")
        upper_label_tab3.config(text="Límite superior:")
        func_label_tab3.config(text="Introducir función objetivo:")
        func_label_tab2.config(text="Introducir función objetivo:")
        numerical_method_label.config(text="Método numérico:")
        history_label.config(text="Historial:")
        result_label_tab1.config(text="")
        result_label_tab2.config(text="")
        result_label_tab3.config(text="")
    elif lang == "Français":
        root.title("Calculateur d’intégrales")
        notebook.tab(0, text="Intégration de base")
        notebook.tab(1, text="Intégration avancée")
        notebook.tab(2, text="Intégrale impropre (infinie)")
        usage_button.config(text="Instructions d’utilisation")
        calc_button_tab1.config(text="Calculer l’intégrale")
        reset_button_tab1.config(text="Réinitialiser")
        calc_button_tab2.config(text="Calculer l’intégrale")
        reset_button_tab2.config(text="Réinitialiser")
        calc_button_tab3.config(text="Calculer l’intégrale")
        reset_button_tab3.config(text="Réinitialiser")
        method_label.config(text="Méthode d’intégration :")
        lower_label_tab2.config(text="Borne inférieure :")
        upper_label_tab2.config(text="Borne supérieure :")
        delta_label_tab2.config(text="Pas (intégration numérique) :")
        func_label_tab2.config(text="Entrer la fonction cible :")
        func_label_tab3.config(text="Entrer la fonction cible :")
        numerical_method_label.config(text="Méthode numérique :")
        history_label.config(text="Historique :")
    elif lang == "العربية":
        root.title("حاسبة التكامل")
        notebook.tab(0, text="تكامل أساسي")
        notebook.tab(1, text="تكامل متقدم")
        notebook.tab(2, text="تكامل غير محدود")
        usage_button.config(text="إرشادات الاستخدام")
        calc_button_tab1.config(text="احسب التكامل")
        reset_button_tab1.config(text="إعادة تعيين")
        calc_button_tab2.config(text="احسب التكامل")
        reset_button_tab2.config(text="إعادة تعيين")
        calc_button_tab3.config(text="احسب التكامل")
        reset_button_tab3.config(text="إعادة تعيين")
        method_label.config(text="طريقة التكامل:")
        lower_label_tab2.config(text="الحد الأدنى:")
        upper_label_tab2.config(text="الحد الأعلى:")
        delta_label_tab2.config(text="حجم الخطوة (تكامل عددي):")
        func_label_tab2.config(text="أدخل الدالة:")
        func_label_tab3.config(text="أدخل الدالة:")
        numerical_method_label.config(text="الطريقة العددية:")
        history_label.config(text="السجل:")
    elif lang == "हिन्दी":
        root.title("समाकलन गणक")
        notebook.tab(0, text="मूल समाकलन")
        notebook.tab(1, text="उन्नत समाकलन")
        notebook.tab(2, text="असंगत समाकलन (अनंत)")
        usage_button.config(text="उपयोग निर्देश")
        calc_button_tab1.config(text="समाकलन गणना करें")
        reset_button_tab1.config(text="रीसेट")
        calc_button_tab2.config(text="समाकलन गणना करें")
        reset_button_tab2.config(text="रीसेट")
        calc_button_tab3.config(text="समाकलन गणना करें")
        reset_button_tab3.config(text="रीसेट")
        method_label.config(text="समाकलन विधि:")
        lower_label_tab2.config(text="निम्न सीमा:")
        upper_label_tab2.config(text="उच्च सीमा:")
        delta_label_tab2.config(text="चरण आकार (संख्यात्मक समाकलन):")
        lower_label_tab3.config(text="निम्न सीमा:")
        upper_label_tab3.config(text="उच्च सीमा:")
        func_label_tab3.config(text="लक्ष्य फलन दर्ज करें:")
        func_label_tab2.config(text="लक्ष्य फलन दर्ज करें:")
        numerical_method_label.config(text="संख्यात्मक विधि:")
        history_label.config(text="इतिहास:")
        result_label_tab1.config(text="")
        result_label_tab2.config(text="")
        result_label_tab3.config(text="")
    update_usage_instructions(lang)

def show_usage_instructions():
    global usage_window
    # If an old window exists but was closed, reset the handle
    if usage_window is not None and not usage_window.winfo_exists():
        usage_window = None
    # If already open, just bring it to front and refresh content
    if usage_window is not None:
        try:
            usage_window.lift()
            update_usage_instructions(lang_var.get())
            return
        except tk.TclError:
            usage_window = None
    # Create new window
    usage_window = tk.Toplevel(root)
    titles = {
        "English": "Usage Instructions",
        "中文（简体）": "使用说明",
        "國語（繁体）": "使用說明",
        "日本語": "使用説明",
        "한국어": "사용 안내",
        "Español": "Instrucciones de uso",
        "Français": "Instructions d’utilisation",
        "العربية": "إرشادات الاستخدام",
        "हिन्दी": "उपयोग निर्देश"
    }
    usage_window.title(titles.get(lang_var.get(), "Usage Instructions"))
    usage_text = "\n".join(instructions.get(lang_var.get(), instructions["English"]))
    usage_label = tk.Label(usage_window, text=usage_text, justify="left", padx=10, pady=10)
    usage_label.pack()
    # Ensure closing the window clears the global reference
    def on_usage_close():
        global usage_window
        if usage_window is not None and usage_window.winfo_exists():
            usage_window.destroy()
        usage_window = None
    usage_window.protocol("WM_DELETE_WINDOW", on_usage_close)

usage_button.config(command=show_usage_instructions)
lang_button.bind("<<ComboboxSelected>>", lambda event: change_language(lang_var.get()))

# ============ Tab 1: Basic Integration ============
def calculate_integral_tab1():
    global last_raw_result
    try:
        func_str = func_entry_tab1.get().strip()
        shown_func = format_function_input(func_str)
        lower_text = lower_entry_tab1.get().strip()
        upper_text = upper_entry_tab1.get().strip()
        expr = evaluate_symbolic_function(func_str)
        if lower_text and upper_text:
            lower = parse_input_to_sympy(lower_text)
            upper = parse_input_to_sympy(upper_text)
            res = integrate(expr, (x_sym, lower, upper))
            fraction_result = nsimplify(res)
            display_str, raw_str = format_result_for_display(fraction_result)
            last_raw_result = raw_str
            result_label_tab1.config(text=f"Definite Integral: {display_str}")
            try:
                l_float = parse_input_to_float(lower_text)
                u_float = parse_input_to_float(upper_text)
                plot_embedded(func_str, l_float, u_float)
            except Exception:
                clear_plot(plot_ax, plot_canvas)
            update_history(f"Definite: ∫[{lower_text}, {upper_text}] {shown_func} dx = {display_str}")
        else:
            ind = integrate(expr, x_sym)
            last_raw_result = sp.sstr(ind)
            result_label_tab1.config(text=f"Indefinite Integral: {last_raw_result} + C")
            update_history(f"Indefinite: ∫ {shown_func} dx = {last_raw_result} + C")
            clear_plot(plot_ax, plot_canvas)
    except ValueError as ve:
        messagebox.showerror("Error", f"{ve}")
    except Exception as e:
        messagebox.showerror("Error", f"Error in integration: {e}")

tk.Label(tab1, text="∫", font=("Arial", 60)).grid(row=0, column=0, rowspan=2, padx=10, pady=5)
upper_entry_tab1 = tk.Entry(tab1, width=10, justify="center"); upper_entry_tab1.grid(row=0, column=1, padx=5, pady=2)
lower_entry_tab1 = tk.Entry(tab1, width=10, justify="center"); lower_entry_tab1.grid(row=1, column=1, padx=5, pady=2)
func_entry_tab1  = tk.Entry(tab1, width=25); func_entry_tab1.grid(row=0, column=2, rowspan=2, padx=5, pady=5)
tk.Label(tab1, text="dx", font=("Arial", 20)).grid(row=0, column=3, rowspan=2, padx=5)
calc_button_tab1 = tk.Button(tab1, text="Calculate Integral", command=calculate_integral_tab1, bg="lightblue")
calc_button_tab1.grid(row=2, column=0, columnspan=4, pady=10)
result_label_tab1 = tk.Label(tab1, text="", fg="green", font=("Arial", 12))
result_label_tab1.grid(row=3, column=0, columnspan=4, pady=10)


# Show exact result popup
def show_exact_result():
    global last_raw_result
    if last_raw_result:
        messagebox.showinfo(
            "Exact Result",
            f"Exact value:\n\n{last_raw_result}"
        )

# Button to view exact result
view_exact_button_tab1 = tk.Button(
    tab1,
    text="View Exact Result",
    command=show_exact_result
)
view_exact_button_tab1.grid(row=4, column=0, columnspan=4, pady=5)

reset_button_tab1 = tk.Button(tab1, text="Reset", bg="lightcoral")
reset_button_tab1.grid(row=5, column=0, columnspan=4, pady=10)

def reset_inputs():
    # Tab1
    func_entry_tab1.delete(0, tk.END); lower_entry_tab1.delete(0, tk.END); upper_entry_tab1.delete(0, tk.END)
    result_label_tab1.config(text="")
    # Tab2
    func_entry_tab2.delete(0, tk.END); lower_entry_tab2.delete(0, tk.END); upper_entry_tab2.delete(0, tk.END)
    delta_entry_tab2.delete(0, tk.END); result_label_tab2.config(text="")
    # Tab3
    func_entry_tab3.delete(0, tk.END); lower_entry_tab3.delete(0, tk.END); upper_entry_tab3.delete(0, tk.END)
    result_label_tab3.config(text="")
    # History
    history.clear(); history_listbox.delete(0, tk.END)
    # Progress & Plot
    progress["value"] = 0; clear_plot(plot_ax, plot_canvas)
    global last_raw_result
    last_raw_result = None

reset_button_tab1.config(command=reset_inputs)

# ============ Tab 2: Advanced ============
event_queue = Queue()
worker_thread = None
worker_running = False

def threaded_calculate_integral_tab2():
    global worker_thread, worker_running
    try:
        func_str = func_entry_tab2.get().strip()
        shown_func = format_function_input(func_str)
        method = method_var.get()
        integration_method = numerical_method_var.get()
        lower_text = lower_entry_tab2.get().strip()
        upper_text = upper_entry_tab2.get().strip()

        if method == "Numerical Integration" and (not lower_text or not upper_text):
            messagebox.showerror("Error", "Numerical integration requires both lower and upper limits.")
            return

        lower = parse_input_to_float(lower_text) if lower_text else None
        upper = parse_input_to_float(upper_text) if upper_text else None
        delta_text = delta_entry_tab2.get().strip()
        delta = None
        if delta_text:
            delta = parse_input_to_float(delta_text)
            if delta <= 0:
                raise ValueError("Step size (delta) must be positive.")

        progress["value"] = 0; progress["maximum"] = 100
        if worker_running:
            messagebox.showinfo("Info", "A computation is already running.")
            return

        def worker():
            global worker_running
            worker_running = True
            try:
                expr = evaluate_symbolic_function(func_str)
                if method == "Symbolic Integration":
                    if lower_text and upper_text:
                        L = parse_input_to_sympy(lower_text)
                        U = parse_input_to_sympy(upper_text)
                        symbolic_result = integrate(expr, (x_sym, L, U))
                        symbolic_result = nsimplify(symbolic_result)
                        formatted = str(symbolic_result).replace('**', '^').replace('pi', 'π')
                        event_queue.put(("symbolic_result", formatted, lower_text, upper_text, shown_func))
                    else:
                        ind = integrate(expr, x_sym)
                        formatted = str(ind).replace('pi', 'π')
                        event_queue.put(("symbolic_indef", formatted, shown_func))
                    for v in range(0, 101, 10):
                        time.sleep(0.03); event_queue.put(("progress", v))
                else:
                    if not np.isfinite(lower) or not np.isfinite(upper):
                        if integration_method in ("Gaussian Quadrature", "Romberg", "Adaptive Simpson",
                                                  "Simpson", "Trapezoidal", "Rectangle", "Simpson 3/8"):
                            raise ValueError(f"{integration_method} requires finite lower and upper limits.")
                    f_num_raw = build_numeric_callable(expr)
                    def f_num(z): return np.asarray(f_num_raw(z), dtype=float)

                    ok, xbad = ensure_finite_on_probe(f_num, lower, upper)
                    if not ok:
                        raise ValueError(f"Function not finite on the interval near x={xbad}. "
                                         f"Consider symbolic integration or splitting the interval.")

                    if integration_method == "Monte Carlo":
                        n_samples = (8000 if delta is None
                                     else max(4000, int(abs(upper-lower)/max(delta,1e-6))*20))
                        n_strata = min(200, max(20, int(np.sqrt(n_samples))))
                        rng = np.random.default_rng(); chunks = 10
                        part_estimates = []
                        for k in range(chunks):
                            est_k = monte_carlo_stratified(
                                f_num, lower, upper,
                                n_samples=max(1, n_samples//chunks),
                                n_strata=max(1, n_strata//chunks),
                                rng=rng
                            )
                            part_estimates.append(est_k)
                            event_queue.put(("progress", int((k+1)*100/chunks)))
                        # 关键：对分块估计取平均，而不是求和
                        result = float(np.mean(part_estimates))

                    elif integration_method == "Rectangle":
                        n = 1000 if delta is None else max(10, int(abs((upper-lower)/delta)))
                        xs = np.linspace(lower, upper, n+1); mids = 0.5*(xs[:-1]+xs[1:])
                        chunks = 20; chunk_size = max(1, len(mids)//chunks); total = 0.0
                        for k in range(chunks):
                            sl = slice(k*chunk_size, (k+1)*chunk_size)
                            xm = mids[sl]; 
                            if xm.size==0: continue
                            fm = f_num(xm).reshape(-1)
                            if not np.all(np.isfinite(fm)): raise ValueError("Function produced non-finite values in Rectangle.")
                            total += np.sum(fm)*(upper-lower)/n
                            event_queue.put(("progress", int((k+1)*100/chunks)))
                        result = total

                    elif integration_method == "Trapezoidal":
                        n = 2000 if delta is None else max(50, int(abs((upper-lower)/delta)))
                        xs = np.linspace(lower, upper, n+1); ys = f_num(xs).reshape(-1)
                        if not np.all(np.isfinite(ys)): raise ValueError("Function produced non-finite values in Trapezoidal.")
                        result = np.trapezoid(ys, xs); event_queue.put(("progress", 100))

                    elif integration_method == "Simpson":
                        n = (2001 if delta is None else max(101, int(abs((upper-lower)/delta))))
                        if n % 2 == 0: n += 1
                        xs = np.linspace(lower, upper, n); ys = f_num(xs).reshape(-1)
                        if not np.all(np.isfinite(ys)): raise ValueError("Function produced non-finite values in Simpson.")
                        result = simpson(ys, x=xs); event_queue.put(("progress", 100))

                    elif integration_method == "Romberg":
                        result = romberg_custom(f_num, lower, upper, max_level=8, tol=1e-8); event_queue.put(("progress", 100))

                    elif integration_method == "Gaussian Quadrature":
                        # 使用自实现 Gauss–Legendre（不依赖弃用 API）
                        result = gaussian_quadrature_fixed(f_num, lower, upper, n=64)
                        event_queue.put(("progress", 100))

                    elif integration_method == "Simpson 3/8":
                        n_int = 300 if delta is None else max(30, int(abs((upper-lower)/delta)))
                        result = composite_simpson_38(lambda t: f_num(t), lower, upper, n_intervals=n_int); event_queue.put(("progress", 100))

                    elif integration_method == "Adaptive Simpson":
                        # 用 quad，但确保标量化返回，避免未来 numpy 行为变化
                        def f_scalar(t):
                            v = np.asarray(f_num(t), dtype=float).reshape(-1)
                            return float(v[0])
                        result, _ = quad(f_scalar, lower, upper, epsabs=1.49e-8, epsrel=1.49e-8, limit=100)
                        event_queue.put(("progress", 100))

                    else:
                        raise ValueError(f"Unknown numerical method: {integration_method}")
                    event_queue.put(("numeric_result", result, integration_method, lower, upper, shown_func, func_str))
            except Exception as e:
                event_queue.put(("error", str(e)))
            finally:
                worker_running = False

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        poll_queue()
    except Exception as e:
        messagebox.showerror("Error", f"Error in integration: {e}")

def poll_queue():
    try:
        while True:
            item = event_queue.get_nowait()
            kind = item[0]
            if kind == "progress":
                progress["value"] = max(progress["value"], int(item[1]))
            elif kind == "symbolic_result":
                formatted, L, U, shown_func = item[1], item[2], item[3], item[4]
                result_label_tab2.config(text=f"Symbolic Integration Result: {formatted}")
                update_history(f"Symbolic: ∫[{L}, {U}] {shown_func} dx = {formatted}")
                try:
                    l_float = parse_input_to_float(L); u_float = parse_input_to_float(U)
                    plot_embedded(shown_func, l_float, u_float)
                except Exception:
                    clear_plot(plot_ax, plot_canvas)
            elif kind == "symbolic_indef":
                formatted, shown_func = item[1], item[2]
                result_label_tab2.config(text=f"Indefinite Integral Result: {formatted} + C")
                update_history(f"Indefinite: ∫ {shown_func} dx = {formatted} + C")
                clear_plot(plot_ax, plot_canvas)
            elif kind == "numeric_result":
                result, method_used, lower, upper, shown_func, raw_func = item[1:]
                formatted_result = f"{result:.12g}"
                result_label_tab2.config(text=f"Numerical Integration Result: {formatted_result}")
                update_history(f"Numerical ({method_used}): ∫[{lower}, {upper}] {shown_func} dx ≈ {formatted_result}")
                try:
                    if np.isfinite(lower) and np.isfinite(upper):
                        plot_embedded(raw_func, lower, upper)
                    else:
                        clear_plot(plot_ax, plot_canvas)
                except Exception:
                    clear_plot(plot_ax, plot_canvas)
            elif kind == "error":
                messagebox.showerror("Error", item[1])
            else:
                pass
    except Exception:
        pass
    if worker_running or progress["value"] < 100:
        root.after(60, poll_queue)

# Tab2 widgets
func_label_tab2 = tk.Label(tab2, text="Enter target function:"); func_label_tab2.grid(row=0, column=0, padx=10, pady=5, sticky='w')
func_entry_tab2 = tk.Entry(tab2, width=30); func_entry_tab2.grid(row=0, column=1, padx=10, pady=5)

lower_label_tab2 = tk.Label(tab2, text="Lower limit:"); lower_label_tab2.grid(row=1, column=0, padx=10, pady=5, sticky='w')
lower_entry_tab2 = tk.Entry(tab2, width=12); lower_entry_tab2.grid(row=1, column=1, padx=10, pady=5, sticky='w')

upper_label_tab2 = tk.Label(tab2, text="Upper limit:"); upper_label_tab2.grid(row=2, column=0, padx=10, pady=5, sticky='w')
upper_entry_tab2 = tk.Entry(tab2, width=12); upper_entry_tab2.grid(row=2, column=1, padx=10, pady=5, sticky='w')

delta_label_tab2 = tk.Label(tab2, text="Step size (for Numerical Integration):"); delta_label_tab2.grid(row=3, column=0, padx=10, pady=5, sticky='w')
delta_entry_tab2 = tk.Entry(tab2, width=12); delta_entry_tab2.grid(row=3, column=1, padx=10, pady=5, sticky='w')

method_label = tk.Label(tab2, text="Integration Method:"); method_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')
method_var = tk.StringVar(value="Symbolic Integration")
method_dropdown = ttk.Combobox(tab2, textvariable=method_var, values=["Symbolic Integration", "Numerical Integration"], state="readonly")
method_dropdown.grid(row=4, column=1, padx=10, pady=5, sticky='w')

numerical_method_label = tk.Label(tab2, text="Numerical Method:"); numerical_method_label.grid(row=5, column=0, padx=10, pady=5, sticky='w')
numerical_method_var = tk.StringVar(value="Trapezoidal")
numerical_method_dropdown = ttk.Combobox(
    tab2, textvariable=numerical_method_var,
    values=["Trapezoidal", "Simpson", "Rectangle", "Romberg", "Gaussian Quadrature", "Simpson 3/8", "Adaptive Simpson", "Monte Carlo"],
    state="readonly"
)
numerical_method_dropdown.grid(row=5, column=1, padx=10, pady=5, sticky='w')

calc_button_tab2 = tk.Button(tab2, text="Calculate Integral", bg="lightblue", command=threaded_calculate_integral_tab2)
calc_button_tab2.grid(row=6, column=0, columnspan=2, pady=10)
reset_button_tab2 = tk.Button(tab2, text="Reset", bg="lightcoral", command=reset_inputs)
reset_button_tab2.grid(row=7, column=0, columnspan=2, pady=10)

result_label_tab2 = tk.Label(tab2, text="", fg="green", font=("Arial", 12))
result_label_tab2.grid(row=8, column=0, columnspan=2, pady=10)

# ============ Tab 3: Improper (Infinite) ============
def compute_general_integral(func_str, lower_text, upper_text):
    f = evaluate_symbolic_function(func_str)
    L = -sp.oo if lower_text == '-inf' else parse_input_to_sympy(lower_text)
    U = sp.oo  if upper_text ==  'inf' else parse_input_to_sympy(upper_text)
    try:
        res = integrate(f, (x_sym, L, U))
        return pretty(res)
    except Exception as e:
        return f"Error: {e}"

def compute_transform():
    try:
        func_str = func_entry_tab3.get().strip()
        shown_func = format_function_input(func_str)
        lower = lower_entry_tab3.get().strip()
        upper = upper_entry_tab3.get().strip()
        if not lower or not upper:
            raise ValueError("Please provide both lower and upper limits (use -inf/inf for infinity).")
        result = compute_general_integral(func_str, lower, upper)
        result_label_tab3.config(text=f"Result: {result}")
        update_history(f"Improper Integral: ∫[{lower}, {upper}] {shown_func} dx = {result}")
        clear_plot(plot_ax, plot_canvas)  # Do not plot infinite intervals
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred in computing the integral: {e}")

func_label_tab3 = tk.Label(tab3, text="Enter target function:"); func_label_tab3.grid(row=0, column=0, padx=10, pady=5, sticky='w')
func_entry_tab3 = tk.Entry(tab3, width=30); func_entry_tab3.grid(row=0, column=1, padx=10, pady=5)
lower_label_tab3 = tk.Label(tab3, text="Lower limit:"); lower_label_tab3.grid(row=1, column=0, padx=10, pady=5, sticky='w')
lower_entry_tab3 = tk.Entry(tab3, width=12); lower_entry_tab3.grid(row=1, column=1, padx=10, pady=5, sticky='w')
upper_label_tab3 = tk.Label(tab3, text="Upper limit:"); upper_label_tab3.grid(row=2, column=0, padx=10, pady=5, sticky='w')
upper_entry_tab3 = tk.Entry(tab3, width=12); upper_entry_tab3.grid(row=2, column=1, padx=10, pady=5, sticky='w')
calc_button_tab3 = tk.Button(tab3, text="Compute Integral", bg="lightblue", command=compute_transform)
calc_button_tab3.grid(row=3, column=0, columnspan=2, pady=10)
reset_button_tab3 = tk.Button(tab3, text="Reset", bg="lightcoral", command=reset_inputs)
reset_button_tab3.grid(row=4, column=0, columnspan=2, pady=10)
result_label_tab3 = tk.Label(tab3, text="", fg="green", font=("Arial", 12))
result_label_tab3.grid(row=5, column=0, columnspan=2, pady=10)

# Apply initial language settings
change_language(lang_var.get())

root.mainloop()