import tkinter as tk
import time
import re
import threading
from tkinter import ttk, messagebox
import math
import numpy as np
from sympy import pi
from sympy import E
from sympy import lambdify,sin,cos,tan,sqrt,exp,log,pi,E
from sympy import asin,acos,atan,asinh,acosh,atanh,sinh,cosh,tanh
from sympy import symbols, integrate, sympify, nsimplify, exp, pretty
import sympy
from scipy.integrate import romberg
from scipy.integrate import quadrature
from scipy.integrate import quad
from scipy.integrate import simpson
import matplotlib.pyplot as plt

def format_function_input(func_str):
    import re
    # 将类似 "2x", "3x^2", "5xy" 等自动添加 *
    formatted_str = re.sub(r'(?<=\d)([a-zA-Z])', r'*\1', func_str)
    return formatted_str
def parse_input(value):
    try:
        value = value.replace('pi', 'sympy.pi').replace('e', 'sympy.E')
        import re
        value = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', value)
        result = eval(value)
        return result
    except Exception as e:
        raise ValueError(f"Invalid input: {e}")


def evaluate_function(x, func_str):
    try:
        # Ensure x is a numpy array for consistent evaluation
        x = np.array(x, dtype=np.float64)
        x_sym = symbols('x')
        
        # Handle expressions like '2pi'
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        
        # Replace 'pi' and 'e' with sympy symbols
        func_str = func_str.replace('pi', 'sympy.pi').replace('e', 'sympy.E')
        
        # Convert string to sympy expression
        func_sympy = sympify(func_str)
        func = lambdify(x_sym, func_sympy, 'numpy')
        result = func(x)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Function evaluation returned invalid values for x = {x}: {result}")
        return result
    except Exception as e:
        raise ValueError(f"Error parsing function: {e}")

def plot_function(func_str, lower, upper):
    try:
        x_vals = np.linspace(lower, upper, 500)
        y_vals = evaluate_function(x_vals, func_str)
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f"f(x) = {func_str}")
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.title(f"Function Graph: {func_str}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.legend()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Error plotting function: {e}")


def evaluate_symbolic_function(func_str):
    try:
        print(f"Original func_str: {func_str}")
        func_str = func_str.replace('^', '**')
        func_str = func_str.replace('ln', 'log')
        func_str = func_str.replace('e', 'sympy.E')
        print(f"Modified func_str: {func_str}")
        func_sympy = sympify(func_str)
        return func_sympy
    except Exception as e:
        raise ValueError(f"Error parsing symbolic function: {e}")

def compute_general_integral(func_str, lower, upper):
    x = symbols('x')
    func_sympy = evaluate_symbolic_function(func_str)

    # Parse lower and upper limits
    if lower == '-inf':
        lower = -np.inf  # SymPy's negative infinity
    else:
        lower = parse_input(lower)

    if upper == 'inf':
        upper = np.inf  # SymPy's positive infinity
    else:
        upper = parse_input(upper)

    try:
        result = integrate(func_sympy, (x, lower, upper))
        # Convert the result to a string for simplicity
        return pretty(result)
    except ValueError as ve:
        return f"Error: {str(ve)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"



history = []

def update_history(new_record):
    # 全局变量history，用于存储历史记录
    global history
    # 将新的记录添加到history列表中
    history.append(new_record)
    # 如果history列表的长度超过5，则移除第一个元素
    if len(history) > 5:
        history.pop(0)
    # 清空history_listbox中的所有项
    history_listbox.delete(0, tk.END)
    # 遍历history列表，将每个记录添加到history_listbox中
    for record in history:
        # 在history_listbox的末尾插入记录
        history_listbox.insert(tk.END, record)

# 重置功能
def reset_inputs():
    # 清空函数输入框
    func_entry_tab1.delete(0, tk.END) 
    # 清空下界输入框
    lower_entry_tab1.delete(0, tk.END) 
    # 清空上界输入框
    upper_entry_tab1.delete(0, tk.END)
    # 清空结果标签
    result_label_tab1.config(text="")

    # 清空函数输入框
    func_entry_tab2.delete(0, tk.END)
    # 清空下界输入框
    lower_entry_tab2.delete(0, tk.END)
    # 清空上界输入框
    upper_entry_tab2.delete(0, tk.END)
    # 清空步长输入框
    delta_entry_tab2.delete(0, tk.END)
    result_label_tab2.config(text="") 

    # 清空函数输入框
    func_entry_tab3.delete(0, tk.END)
    # 清空下界输入框
    lower_entry_tab3.delete(0, tk.END)
    # 清空上界输入框
    upper_entry_tab3.delete(0, tk.END)
    # 清空结果标签
    result_label_tab3.config(text="")

    # 清空历史记录列表
    history.clear()
    # 清空历史记录列表框
    history_listbox.delete(0, tk.END)
# 使用说明
instructions = {
    "English": [
        "Common mathematical functions and constants:",
        "log10(x) - Common logarithm (base 10), e.g., log10(100)",
        "ln(x) - Natural logarithm (base e), e.g., ln(2)",
        "pi - Pi, input: pi",
        "sin(x) - Sine function, e.g., sin(pi / 4)",
        "cos(x) - Cosine function, e.g., cos(pi / 3)",
        "tan(x) - Tangent function, e.g., tan(pi / 4)",
        "sqrt(x) - Square root, e.g., sqrt(4)",
        "exp(x) - Exponential function, e.g., exp(1)"
        "To represent infinity:",
        "Positive infinity - inf",
        "Negative infinity - -inf",
        "",
        "日本語",
        "一般的な数学関数と定数の使用方法：",
        "log10(x) - 常用対数（底10）、例：log10(100)",
        "ln(x) - 自然対数（底e）、例：ln(2)",
        "pi - 円周率，入力：pi",
        "sin(x) - 正弦関数、例：sin(pi / 4)",
        "cos(x) - 余弦関数、例：cos(pi / 3)",
        "tan(x) - 正接関数、例：tan(pi / 4)",
        "sqrt(x) - 平方根、例：sqrt(4)",
        "exp(x) - 指数関数、例：exp(1)",
        "無限大を表す方法：",
        "正の無限大 - inf",
        "負の無限大 - -inf"
        "",
        "",
        "国文",
        "常见数学函数和常数的使用方法：",
        "log10(x) - 常用对数（以10为底），例如：log10(100)",
        "ln(x) - 自然对数（以e为底），例如：ln(2)",
        "pi - 圆周率，输入：pi",
        "sin(x) - 正弦函数，例如：sin(pi / 4)",
        "cos(x) - 余弦函数，例如：cos(pi / 3)",
        "tan(x) - 正切函数，例如：tan(pi / 4)",
        "sqrt(x) - 平方根，例如：sqrt(4)",
        "exp(x) - 指数函数，例如：exp(1)",
        "表示无穷大的方法：",
        "正无穷大 - inf",
        "负无穷大 - -inf",

    ]
}




# 创建主窗口
root = tk.Tk()
root.title("Integration Calculator")

# 定义语言变量
lang_var = tk.StringVar(value="English")
usage_window = None  # 定义全局变量

def change_language(lang):
    if lang == "English":
        root.title("Integration Calculator")
        notebook.tab(0, text="Basic Integration")
        notebook.tab(1, text="Advanced Integration")
        notebook.tab(2, text="Improper Integral (Infinite)")
        usage_button.config(text="Usage Instructions")
        lang_button.set("Language")
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
    elif lang == "中文":
        root.title("积分计算器")
        notebook.tab(0, text="基本积分")
        notebook.tab(1, text="进阶积分")
        notebook.tab(2, text="反常积分（无穷）")
        usage_button.config(text="使用说明")
        lang_button.set("语言")
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
    elif lang == "日本語":
        root.title("積分計算機")
        notebook.tab(0, text="基本積分")
        notebook.tab(1, text="高度な積分")
        notebook.tab(2, text="異常積分（無限）")
        usage_button.config(text="使用説明")
        lang_button.set("言語")
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

    # 调用 update_usage_instructions 更新使用说明文本
    update_usage_instructions(lang)

def update_usage_instructions(lang):
    usage_text = "\n".join(instructions.get(lang, instructions["English"]))
    if usage_window is not None:
        for widget in usage_window.winfo_children():
            widget.destroy()
        usage_label = tk.Label(usage_window, text=usage_text, justify="left", padx=10, pady=10)
        usage_label.pack()

def show_usage_instructions():
    # 定义全局变量 usage_window
    global usage_window
    
    # 如果 usage_window 已存在，则销毁它
    if usage_window is not None:
        usage_window.destroy()
    
    # 创建一个新的 Toplevel 窗口作为 usage_window
    usage_window = tk.Toplevel(root)
    
    # 设置 usage_window 的标题为“使用说明”
    usage_window.title("使用说明")
    
    # 获取当前选中的语言
    selected_lang = lang_var.get()
    
    # 根据选中的语言获取相应的使用说明文本
    usage_text = "\n".join(instructions.get(selected_lang, instructions["English"]))
    
    # 创建一个 Label 组件显示使用说明文本
    usage_label = tk.Label(usage_window, text=usage_text, justify="left", padx=10, pady=10)
    
    # 将 Label 组件添加到 usage_window 中
    usage_label.pack()
lang_button = ttk.Combobox(root, textvariable=lang_var, values=["English", "中文", "日本語"], state="readonly")
lang_button.pack(pady=10)
lang_button.bind("<<ComboboxSelected>>", lambda event: change_language(lang_var.get()))

# 显示使用说明窗口按钮
usage_button = tk.Button(root, text="Usage Instructions", command=show_usage_instructions, bg="lightgreen")
usage_button.pack(pady=10)

# 创建标签页
notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
tab3 = ttk.Frame(notebook)
notebook.add(tab1, text="Basic Integration")
notebook.add(tab2, text="Advanced Integration")
notebook.add(tab3, text="Improper Integral (Infinite)")
notebook.pack(expand=1, fill="both")

# 第一个标签页内容
def calculate_integral_tab1():
    try:
        func_str = func_entry_tab1.get()
        func_str = format_function_input(func_str)
        lower_text = lower_entry_tab1.get().strip()
        upper_text = upper_entry_tab1.get().strip()

        x = symbols('x')
        func_sympy = evaluate_symbolic_function(func_str)
        
        if lower_text and upper_text:
            lower = parse_input(lower_text)
            upper = parse_input(upper_text)
            result = integrate(func_sympy, (x, lower, upper))
            fraction_result = nsimplify(result)
            formatted_result = str(fraction_result).replace('**', '^').replace('E', 'e').replace('exp', 'e^').replace('pi', 'π')
            result_label_tab1.config(text=f"Definite Integral: {formatted_result}")
            plot_function(func_str, lower, upper)
            update_history(f"Definite: ∫[{lower}, {upper}] {func_str} dx = {formatted_result}")
        else:
            indefinite_result = integrate(func_sympy, x)
            indefinite_result = str(indefinite_result).replace('pi', 'π')
            result_label_tab1.config(text=f"Indefinite Integral: {indefinite_result} + C")
            update_history(f"Indefinite: ∫{func_str} dx = {indefinite_result} + C")
    except ValueError as ve:
        messagebox.showerror("Error", f"{ve}")
    except Exception as e:
        messagebox.showerror("Error", f"Error in integration: {e}")


tk.Label(tab1, text="∫", font=("Arial", 60)).grid(row=0, column=0, rowspan=2, padx=10, pady=5)
upper_entry_tab1 = tk.Entry(tab1, width=7, justify="center")
upper_entry_tab1.grid(row=0, column=1, padx=5, pady=2)
lower_entry_tab1 = tk.Entry(tab1, width=7, justify="center")
lower_entry_tab1.grid(row=1, column=1, padx=5, pady=2)
func_entry_tab1 = tk.Entry(tab1, width=15)
func_entry_tab1.grid(row=0, column=2, rowspan=2, padx=5, pady=5)
tk.Label(tab1, text="dx", font=("Arial", 20)).grid(row=0, column=3, rowspan=2, padx=5)
calc_button_tab1 = tk.Button(tab1, text="Calculate Integral", command=calculate_integral_tab1, bg="lightblue")
calc_button_tab1.grid(row=2, column=0, columnspan=4, pady=10)
result_label_tab1 = tk.Label(tab1, text="", fg="green", font=("Arial", 12))
result_label_tab1.grid(row=3, column=0, columnspan=4, pady=10)
reset_button_tab1 = tk.Button(tab1, text="Reset", command=reset_inputs, bg="lightcoral")
reset_button_tab1.grid(row=4, column=0, columnspan=4, pady=10)


# 第二个标签页内容
# 创建一个进度条
progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=200, mode='determinate')
progress.pack(pady=10)

# 第二个标签页内容
def threaded_calculate_integral_tab2():
    try:
        # 获取用户输入的函数表达式
        func_str = func_entry_tab2.get()
        func_str = format_function_input(func_str)
        lower_text = lower_entry_tab2.get().strip()
        upper_text = upper_entry_tab2.get().strip()
        method = method_var.get()
        integration_method = numerical_method_var.get()

        if method == "Numerical Integration":
            if not lower_text or not upper_text:
                messagebox.showerror("Error", "Numerical integration requires both lower and upper limits.")
                return

        lower = parse_input(lower_text)
        upper = parse_input(upper_text)
        delta = parse_input(delta_entry_tab2.get())

        # 重置进度条
        progress["value"] = 0
        progress["maximum"] = 100

        # 启动线程执行积分计算
        thread = threading.Thread(target=perform_integration, args=(func_str, lower_text, upper_text, method, integration_method, delta))
        thread.start()
    except Exception as e:
        messagebox.showerror("Error", f"Error in integration: {e}")

def perform_integration(func_str, lower_text, upper_text, method, integration_method, delta):
    try:
        lower = parse_input(lower_text)
        upper = parse_input(upper_text)
        func = lambda x: evaluate_function(x, func_str)
        x_vals = np.arange(lower, upper, delta)

        result = None
        progress_step = 100 / len(x_vals)

        if method == "Numerical Integration":
            current_result = 0  # 临时变量用于实时显示
            for i, x in enumerate(x_vals):
                # 添加小延迟以减慢进度条更新
                time.sleep(0.02)  # 调整此值以控制速度
                if integration_method == "Trapezoidal":
                    result = np.trapezoid([func(x) for x in x_vals], x_vals)
                elif integration_method == "Simpson":
                    y_vals = [func(x) for x in x_vals]
                    result = simpson(y_vals, x=x_vals) # Ensure correct argument usage
                elif integration_method == "Rectangle":
                    result = np.sum([func(x) * delta for x in x_vals])
                elif integration_method == "Romberg":
                    result, _ = quad(func, lower, upper)
                
                elif integration_method == "Gaussian Quadrature":
                    # 确保 lower 和 upper 是数值类型
                    lower = np.float64(lower)
                    upper = np.float64(upper)
                    
                    # 检查 lower 和 upper 是否为有限的数字
                    if not (np.isfinite(lower) and np.isfinite(upper)):
                        raise ValueError("Gaussian Quadrature method requires finite lower and upper limits.")
                    
                    # 检查 func 是否可以正确评估
                    test_val = func((lower + upper) / 2)
                    if not np.isfinite(test_val):
                        raise ValueError("The function evaluated at the midpoint is not finite.")
                    
                    result, _ = quadrature(func, lower, upper)


                elif integration_method == "Simpson 3/8":
                    # 确保 lower 和 upper 是数值类型
                    lower = np.float64(lower)
                    upper = np.float64(upper)
                    
                    # 检查 lower 和 upper 是否为有限的数字
                    if not (np.isfinite(lower) and np.isfinite(upper)):
                        raise ValueError("Simpson 3/8 method requires finite lower and upper limits.")
                    
                    # 检查 func 是否可以正确评估
                    test_val = func((lower + upper) / 2)
                    if not np.isfinite(test_val):
                        raise ValueError("The function evaluated at the midpoint is not finite.")
                    
                    result = romberg(func, lower, upper)

                elif integration_method == "Adaptive Simpson":
                    result, _ = quad(func, lower, upper, epsabs=1.49e-08, epsrel=1.49e-08, limit=50)
                elif integration_method == "Monte Carlo":
                    sample_points = np.random.uniform(lower, upper, int(delta))
                    result = (upper - lower) * np.mean([func(x) for x in sample_points])

                progress["value"] += progress_step
            
        elif method == "Symbolic Integration":
                x = symbols('x')
                func_sympy = evaluate_symbolic_function(func_str)
                if lower_text and upper_text:
                    lower = parse_input(lower_text)

                    upper = parse_input(upper_text)
                    symbolic_result = integrate(func_sympy, (x, lower, upper))
                    symbolic_result = nsimplify(symbolic_result)
                    formatted_result = f"{symbolic_result}"
                    formatted_result = str(formatted_result).replace('exp', 'e^',10).replace('**', '^',10).replace('pi','π').replace('(','').replace(')','')
                    result_label_tab2.config(text=f"Symbolic Integration Result: {formatted_result}")
                    update_history(f"Symbolic: ∫[{lower}, {upper}] {func_str} dx = {formatted_result}")
                else:
                    indefinite_result = integrate(func_sympy, x)
                    result_label_tab2.config(text=f"Indefinite Integral Result: {indefinite_result} + C")
                    update_history(f"Indefinite: ∫{func_str} dx = {indefinite_result} + C")
        progress["value"] = 100
        if result is not None:
                formatted_result = str(result).replace('**', '^').replace('e', 'sympy.E').replace('exp', 'e^')
                result_label_tab2.config(text=f"Numerical Integration Result: {formatted_result}")
                update_history(f"Numerical ({integration_method}): ∫[{lower}, {upper}] {func_str} dx ≈ {formatted_result}")
        root.after(0, plot_function, func_str, lower, upper)

    except Exception as e:
        messagebox.showerror("Error", f"Error in integration: {e}")


func_label_tab2 = tk.Label(tab2, text="Enter target function:")
func_label_tab2.grid(row=0, column=0, padx=10, pady=5, sticky='w')
func_entry_tab2 = tk.Entry(tab2, width=30)
func_entry_tab2.grid(row=0, column=1, padx=10, pady=5)
lower_label_tab2 = tk.Label(tab2, text="Lower limit:")
lower_label_tab2.grid(row=1, column=0, padx=10, pady=5, sticky='w')
lower_entry_tab2 = tk.Entry(tab2, width=10)
lower_entry_tab2.grid(row=1, column=1, padx=10, pady=5, sticky='w')
upper_label_tab2 = tk.Label(tab2, text="Upper limit:")
upper_label_tab2.grid(row=2, column=0, padx=10, pady=5, sticky='w')
upper_entry_tab2 = tk.Entry(tab2, width=10)
upper_entry_tab2.grid(row=2, column=1, padx=10, pady=5, sticky='w')
delta_label_tab2 = tk.Label(tab2, text="Step size (for Numerical Integration):")
delta_label_tab2.grid(row=3, column=0, padx=10, pady=5, sticky='w')
delta_entry_tab2 = tk.Entry(tab2, width=10)
delta_entry_tab2.grid(row=3, column=1, padx=10, pady=5, sticky='w')

method_label = tk.Label(tab2, text="Integration Method:")
method_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')
method_var = tk.StringVar(value="Symbolic Integration")
method_dropdown = ttk.Combobox(tab2, textvariable=method_var, values=["Symbolic Integration", "Numerical Integration"], state="readonly")
method_dropdown.grid(row=4, column=1, padx=10, pady=5, sticky='w')

numerical_method_label = tk.Label(tab2, text="Numerical Method:")
numerical_method_label.grid(row=5, column=0, padx=10, pady=5, sticky='w')
numerical_method_var = tk.StringVar(value="Trapezoidal")
numerical_method_dropdown = ttk.Combobox(tab2, textvariable=numerical_method_var, values=["Trapezoidal", "Simpson", "Rectangle", "Romberg", "Gaussian Quadrature", "Simpson 3/8", "Adaptive Simpson", "Monte Carlo"], state="readonly")

numerical_method_dropdown.grid(row=5, column=1, padx=10, pady=5, sticky='w')

calc_button_tab2 = tk.Button(tab2, text="Calculate Integral", command=threaded_calculate_integral_tab2, bg="lightblue")
calc_button_tab2.grid(row=6, column=0, columnspan=2, pady=10)
reset_button_tab2 = tk.Button(tab2, text="Reset", command=reset_inputs, bg="lightcoral")
reset_button_tab2.grid(row=7, column=0, columnspan=2, pady=10)

result_label_tab2 = tk.Label(tab2, text="", fg="green", font=("Arial", 12))
result_label_tab2.grid(row=8, column=0, columnspan=2, pady=10)

def compute_transform():
    try:
        func_str = func_entry_tab3.get()
        func_str = format_function_input(func_str)
        lower = lower_entry_tab3.get().strip()
        upper = upper_entry_tab3.get().strip()
        result = compute_general_integral(func_str, lower, upper)
        
        # Display the result or error message
        result_label_tab3.config(text=f"Result: {result}")
        update_history(f"Improper Integral: ∫[{lower}, {upper}] {func_str} dx = {result}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred in computing the integral: {e}")

func_label_tab3 = tk.Label(tab3, text="Enter target function:")
func_label_tab3.grid(row=0, column=0, padx=10, pady=5, sticky='w')
func_entry_tab3 = tk.Entry(tab3, width=30)
func_entry_tab3.grid(row=0, column=1, padx=10, pady=5)

lower_label_tab3 = tk.Label(tab3, text="Lower limit:")
lower_label_tab3.grid(row=1, column=0, padx=10, pady=5, sticky='w')
lower_entry_tab3 = tk.Entry(tab3, width=10)
lower_entry_tab3.grid(row=1, column=1, padx=10, pady=5, sticky='w')
upper_label_tab3 = tk.Label(tab3, text="Upper limit:")
upper_label_tab3.grid(row=2, column=0, padx=10, pady=5, sticky='w')
upper_entry_tab3 = tk.Entry(tab3, width=10)
upper_entry_tab3.grid(row=2, column=1, padx=10, pady=5, sticky='w')

calc_button_tab3 = tk.Button(tab3, text="Compute Integral", command=compute_transform, bg="lightblue")
calc_button_tab3.grid(row=3, column=0, columnspan=2, pady=10)
reset_button_tab3 = tk.Button(tab3, text="Reset", command=reset_inputs, bg="lightcoral")
reset_button_tab3.grid(row=4, column=0, columnspan=2, pady=10)

result_label_tab3 = tk.Label(tab3, text="", fg="green", font=("Arial", 12))
result_label_tab3.grid(row=5, column=0, columnspan=2, pady=10)

# 显示历史记录
history_frame = ttk.Frame(root)
history_frame.pack(fill="both", expand=True, padx=10, pady=10)

history_label = tk.Label(history_frame, text="History:", font=("Arial", 12, "bold"))
history_label.pack(anchor="w")
history_text = tk.StringVar(value="")
history_listbox = tk.Listbox(history_frame, height=5, width=70)
history_listbox.pack(fill="x", expand=True, pady=5)
root.mainloop()
