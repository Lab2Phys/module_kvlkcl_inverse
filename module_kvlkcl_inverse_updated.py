import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate
from collections import defaultdict, deque
import itertools
import ipywidgets as widgets
from IPython.display import clear_output

# --- بخش محاسبات ---

def precompute_system(loops, edges):
    """
    ماتریس امپدانس Z، گراف مدار و مپ مقاومت‌ها را پیش‌محاسبه می‌کند.
    """
    n = len(loops)
    Z = np.zeros((n, n), dtype=np.float64)
    all_nodes = set(n for edge in edges for n in edge[:2])
    
    # محاسبه مقاومت کل هر حلقه (قطر اصلی ماتریس Z)
    for i in range(n):
        total_resistance = 0
        loop_nodes = loops[i]
        loop_edges_path = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
        for n1, n2 in loop_edges_path:
            for u, v, r in edges:
                if (u, v) == (n1, n2) or (u, v) == (n2, n1):
                    total_resistance += r
                    break
        Z[i, i] = total_resistance

    # محاسبه مقاومت‌های مشترک بین حلقه‌ها
    for u_edge, v_edge, r_edge in edges:
        if r_edge == 0: 
            continue
            
        loops_containing_edge = []
        for i, loop_nodes in enumerate(loops):
            path_edges = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
            for n1, n2 in path_edges:
                if (n1, n2) == (u_edge, v_edge):
                    loops_containing_edge.append((i, 1))
                    break
                elif (n1, n2) == (v_edge, u_edge):
                    loops_containing_edge.append((i, -1))
                    break
        
        # برای هر جفت حلقه که این یال مشترک دارند
        for idx1 in range(len(loops_containing_edge)):
            for idx2 in range(idx1 + 1, len(loops_containing_edge)):
                i, dir_i = loops_containing_edge[idx1]
                j, dir_j = loops_containing_edge[idx2]
                # اگر جهت جریان‌ها یکسان باشد، مقاومت مشترک منفی است
                Z[i, j] -= r_edge * dir_i * dir_j
                Z[j, i] -= r_edge * dir_i * dir_j

    # ایجاد گراف و نقشه مقاومت‌ها
    graph = defaultdict(list)
    resistance_map = {}
    for u, v, r in edges:
        graph[u].append(v)
        graph[v].append(u)
        resistance_map[(u, v)] = resistance_map[(v, u)] = r
        
    return Z, graph, resistance_map, sorted(list(all_nodes))

def calculate_potentials(unknown_v_value, voltage_source_config, loops, Z_matrix, graph, resistance_map):
    """
    با داشتن مقدار منبع مجهول، جریان‌ها و پتانسیل تمام گره‌ها را محاسبه می‌کند.
    """
    n = len(loops)
    vs = {}
    
    # پیدا کردن منبع مجهول و تعیین مقدار آن
    for (u, v), val in voltage_source_config.items():
        if val == 1 or val == -1:
            vs[(u, v)] = val * unknown_v_value
        else:
            vs[(u, v)] = val

    # محاسبه بردار ولتاژ حلقه‌ها
    V = np.zeros((n, 1), dtype=np.float64)
    for i, loop_nodes in enumerate(loops):
        loop_voltage = 0
        current_edges = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
        for u, v in current_edges:
            loop_voltage += vs.get((u, v), 0)
            loop_voltage -= vs.get((v, u), 0)
        V[i] = loop_voltage

    # حل دستگاه معادلات برای یافتن جریان حلقه‌ها
    I = np.linalg.solve(Z_matrix, V)
    
    # محاسبه جریان هر شاخه
    branch_currents = defaultdict(float)
    for i, loop_nodes in enumerate(loops):
        loop_current = I[i, 0]
        current_edges = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
        for u, v in current_edges:
            branch_currents[(u, v)] += loop_current
            
    # انتخاب گره مرجع (بالاترین شماره گره)
    ref_node = max(graph.keys())
    potentials = {ref_node: 0.0}
    queue = deque([ref_node])
    visited = {ref_node}
    
    # محاسبه پتانسیل سایر گره‌ها با روش پیمایش گراف
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in visited:
                # جریان خالص از u به v
                i_uv = branch_currents.get((u, v), 0.0) - branch_currents.get((v, u), 0.0)
                # افت ولتاژ مقاومتی
                v_drop_r = i_uv * resistance_map.get((u, v), 0.0)
                # افزایش ولتاژ به دلیل منابع ولتاژ
                v_gain_e = vs.get((v, u), 0.0) - vs.get((u, v), 0.0)
                potentials[v] = potentials[u] + v_drop_r + v_gain_e
                visited.add(v)
                queue.append(v)
                
    return potentials, branch_currents

def find_unknown_voltage(node_a, node_b, signed_voltage, **kwargs):
    """
    مقدار منبع ولتاژ مجهول را بر اساس اختلاف پتانسیل معلوم بین دو گره پیدا می‌کند.
    """
    # محاسبه پتانسیل‌ها برای دو مقدار مختلف منبع مجهول
    pot0, _ = calculate_potentials(0.0, **kwargs)
    pot1, _ = calculate_potentials(1.0, **kwargs)
    
    # اختلاف پتانسیل بین گره‌های مورد نظر
    v0 = pot0[node_a] - pot0[node_b]
    v1 = pot1[node_a] - pot1[node_b]
    k = v1 - v0
    
    if abs(k) < 1e-12:
        raise ValueError("This node pair is not sensitive to the unknown source! Try different nodes.")
        
    # حل معادله خطی برای یافتن مقدار منبع مجهول
    return (signed_voltage - v0) / k

def make_voltage_table(potentials, all_nodes, decimal_places):
    """جدول ولتاژها بین تمام جفت گره‌ها"""
    return [
        [f"({n1}, {n2})", f"{abs(potentials[n2] - potentials[n1]):.{decimal_places}f}"]
        for n1, n2 in itertools.combinations(all_nodes, 2)
    ]

def make_current_table(branch_currents, edges, decimal_places):
    """جدول جریان‌های شاخه‌ها برحسب میلی‌آمپر"""
    table_data = []
    processed_edges = set()
    
    for u, v, R in edges:
        if (u, v) in processed_edges or (v, u) in processed_edges:
            continue
        
        # جریان خالص از u به v
        net_current_uv = branch_currents.get((u, v), 0.0) - branch_currents.get((v, u), 0.0)
        
        # تبدیل به میلی‌آمپر
        current_mA = net_current_uv * 1000
        direction = f"{u} → {v}" if current_mA >= 0 else f"{v} → {u}"
        table_data.append([direction, f"{abs(current_mA):.{decimal_places}f}"])
        processed_edges.add((u, v))
        
    return table_data

# --- بخش خروجی و PDF ---

def add_table_to_pdf(pdf, table_data, title, col_labels, col_widths=[0.3, 0.3]):
    """افزودن جدول استایل‌شده به PDF"""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # سایز A4
    ax.axis('off')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    t = ax.table(cellText=table_data, colLabels=col_labels, loc='center', colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1.2, 1.8)
    
    for (row, col), cell in t.get_celld().items():
        cell.set_text_props(ha='center', va='center')
        if row == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')
            
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def print_boxed_result(unknown_v, decimal_places, unknown_name="Unknown V"):
    """نمایش نتیجه در قاب"""
    text = f"Calculated value: {unknown_name} = {unknown_v:.{decimal_places}f} V"
    width = len(text)
    print(f"┌{'─' * (width + 2)}┐")
    print(f"│ {text} │")
    print(f"└{'─' * (width + 2)}┘")

# --- تابع اصلی سازنده رابط کاربری ---

def create_analyzer_ui(loops, edges, voltage_source_config, decimal_places, unknown_name="Unknown V"):
    """
    تابع اصلی برای ایجاد رابط کاربری تحلیلگر مدار
    """
    # پیش‌محاسبات
    Z_matrix, graph, resistance_map, all_nodes = precompute_system(loops, edges)
    num_loops = len(loops)
    num_nodes = len(all_nodes)
    
    # تشخیص منبع مجهول
    unknown_source = None
    for (u, v), val in voltage_source_config.items():
        if val == 1 or val == -1:
            unknown_source = (u, v, val)
            break
    
    if unknown_source is None:
        raise ValueError("No unknown voltage source found! Use coefficient +1 or -1 for unknown source.")
    
    # ایجاد ویجت‌ها
    node_pair_options = [(f"({p[0]}, {p[1]})", p) for p in itertools.combinations(all_nodes, 2)]
    node_pair_dropdown = widgets.Dropdown(
        options=node_pair_options,
        description='Node Pair:',
        layout=widgets.Layout(width='300px')
    )
    
    voltage_magnitude_box = widgets.FloatText(
        value=0.01, 
        description='|Voltage| (V):', 
        layout=widgets.Layout(width='250px'),
        step=0.001
    )
    
    voltage_sign_dropdown = widgets.Dropdown(
        options=[('V₁ > V₂ (+)', '+'), ('V₂ > V₁ (-)', '-'), ('Check both cases', 'both')],
        value='both', 
        description='Direction:', 
        layout=widgets.Layout(width='300px')
    )
    
    calc_button = widgets.Button(
        description="Calculate", 
        button_style='success', 
        layout=widgets.Layout(width='150px')
    )
    
    output_area = widgets.Output()

    # تابع مدیریت کلیک دکمه
    def on_calculate_clicked(b):
        with output_area:
            clear_output(wait=True)
            try:
                n1, n2 = node_pair_dropdown.value
                v_magnitude = abs(voltage_magnitude_box.value)
                sign_choice = voltage_sign_dropdown.value

                if n1 not in all_nodes or n2 not in all_nodes or n1 == n2:
                    print("❌ Error: Please select a valid node pair.")
                    return

                calc_args = {
                    "voltage_source_config": voltage_source_config, 
                    "loops": loops, 
                    "Z_matrix": Z_matrix, 
                    "graph": graph, 
                    "resistance_map": resistance_map
                }

                print(f"Circuit Analysis: {num_nodes} nodes, {num_loops} loops")
                print(f"Unknown source: {unknown_source[0]} → {unknown_source[1]} (coefficient: {unknown_source[2]:+d})")
                print("-" * 60)

                if sign_choice == 'both':
                    print(f"\n🔍 Checking both possible cases for nodes ({n1}, {n2}):\n" + "="*60)
                    
                    with PdfPages('Voltages.pdf') as pdf_v, PdfPages('Currents.pdf') as pdf_c:
                        # حالت اول: V(n1) > V(n2)
                        print(f"\n📊 Case 1: V({n1}) > V({n2}), ΔV = +{v_magnitude:.{decimal_places}f} V")
                        v_pos = find_unknown_voltage(n1, n2, v_magnitude, **calc_args)
                        pot_pos, b_currents_pos = calculate_potentials(v_pos, **calc_args)
                        table_v_pos = make_voltage_table(pot_pos, all_nodes, decimal_places)
                        table_c_pos = make_current_table(b_currents_pos, edges, decimal_places)
                        
                        print_boxed_result(v_pos, decimal_places, unknown_name)
                        print("\n📋 Voltage Table:")
                        print(tabulate(table_v_pos, headers=["Nodes", "Voltage (V)"], tablefmt="fancy_grid"))
                        
                        # حالت دوم: V(n2) > V(n1)
                        print(f"\n📊 Case 2: V({n2}) > V({n1}), ΔV = -{v_magnitude:.{decimal_places}f} V")
                        v_neg = find_unknown_voltage(n1, n2, -v_magnitude, **calc_args)
                        pot_neg, b_currents_neg = calculate_potentials(v_neg, **calc_args)
                        table_v_neg = make_voltage_table(pot_neg, all_nodes, decimal_places)
                        table_c_neg = make_current_table(b_currents_neg, edges, decimal_places)
                        
                        print_boxed_result(v_neg, decimal_places, unknown_name)
                        print("\n📋 Voltage Table:")
                        print(tabulate(table_v_neg, headers=["Nodes", "Voltage (V)"], tablefmt="fancy_grid"))
                        
                        # ذخیره در PDF
                        add_table_to_pdf(pdf_v, table_v_pos, f"Voltage Table for {unknown_name} = {v_pos:.{decimal_places}f} V", 
                                       ["Nodes", "Voltage (V)"])
                        add_table_to_pdf(pdf_c, table_c_pos, f"Current Table for {unknown_name} = {v_pos:.{decimal_places}f} V", 
                                       ["Branch", "Current (mA)"])
                        
                        if abs(v_pos - v_neg) > 1e-9:
                            add_table_to_pdf(pdf_v, table_v_neg, f"Voltage Table for {unknown_name} = {v_neg:.{decimal_places}f} V", 
                                           ["Nodes", "Voltage (V)"])
                            add_table_to_pdf(pdf_c, table_c_neg, f"Current Table for {unknown_name} = {v_neg:.{decimal_places}f} V", 
                                           ["Branch", "Current (mA)"])

                    print(f"\n✅ Results saved in 'Voltages.pdf' and 'Currents.pdf'.")

                else:  # محاسبه تک حالته
                    signed_voltage = v_magnitude if sign_choice == '+' else -v_magnitude
                    n_high, n_low = (n1, n2) if sign_choice == '+' else (n2, n1)
                    print(f"\n📊 Selected case: V({n_high}) > V({n_low}), ΔV = {signed_voltage:.{decimal_places}f} V")
                    
                    v_result = find_unknown_voltage(n1, n2, signed_voltage, **calc_args)
                    potentials, branch_currents = calculate_potentials(v_result, **calc_args)
                    table_v = make_voltage_table(potentials, all_nodes, decimal_places)
                    table_c = make_current_table(branch_currents, edges, decimal_places)
                    
                    print_boxed_result(v_result, decimal_places, unknown_name)
                    print("\n📋 Voltage Table:")
                    print(tabulate(table_v, headers=["Nodes", "Voltage (V)"], tablefmt="fancy_grid"))
                    print("\n⚡ Current Table:")
                    print(tabulate(table_c, headers=["Branch", "Current (mA)"], tablefmt="fancy_grid"))
                    
                    with PdfPages('Voltages.pdf') as pdf_v, PdfPages('Currents.pdf') as pdf_c:
                        add_table_to_pdf(pdf_v, table_v, f"Voltage Table for {unknown_name} = {v_result:.{decimal_places}f} V", 
                                       ["Nodes", "Voltage (V)"])
                        add_table_to_pdf(pdf_c, table_c, f"Current Table for {unknown_name} = {v_result:.{decimal_places}f} V", 
                                       ["Branch", "Current (mA)"])
                    
                    print("\nResults saved to 'Voltages.pdf' and 'Currents.pdf'.")

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                print(f"Debug info: {traceback.format_exc()}")

    calc_button.on_click(on_calculate_clicked)

    # چیدمان رابط کاربری
    ui_layout = widgets.VBox([
        widgets.HTML(value=f"<h3>🔧 Circuit Analyzer ({num_nodes} nodes, {num_loops} loops)</h3>"),
        widgets.HTML(value=f"<p><b>Unknown source:</b> {unknown_source[0]} → {unknown_source[1]} (coeff: {unknown_source[2]:+d})</p>"),
        widgets.HBox([node_pair_dropdown, voltage_magnitude_box]),
        widgets.HBox([voltage_sign_dropdown, calc_button]),
        widgets.HTML(value="<hr><h3>📊 Results</h3>"),
        output_area
    ])
    
    return ui_layout