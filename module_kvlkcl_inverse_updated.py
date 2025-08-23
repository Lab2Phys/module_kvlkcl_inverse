import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate
from collections import defaultdict, deque
import itertools
import ipywidgets as widgets
from IPython.display import clear_output
from scipy.optimize import minimize_scalar

# --- بخش محاسبات ---

def precompute_system_v2(n, loops, edges, voltage_source_config):
    """
    سیستم معادلات را بر اساس قانون ولتاژ کیرشهف (KVL) پیش‌محاسبه می‌کند.
    ماتریس Z شامل مقاومت‌های داخلی و متقابل حلقه‌ها است.
    """
    Z = np.zeros((n, n), dtype=np.float64)
    # ساخت گراف و نقشه مقاومت‌ها
    graph = defaultdict(list)
    resistance_map = {}
    for u, v, r in edges:
        graph[u].append(v)
        graph[v].append(u)
        resistance_map[(u, v)] = resistance_map[(v, u)] = r

    # محاسبه مقاومت‌های داخلی (Zii)
    for i, loop_nodes in enumerate(loops):
        total_resistance = 0
        loop_edges_path = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
        for n1, n2 in loop_edges_path:
            total_resistance += resistance_map.get((n1, n2), 0)
        Z[i, i] = total_resistance

    # محاسبه مقاومت‌های متقابل (Zij)
    for u_edge, v_edge, r_edge in edges:
        loops_containing_edge = []
        for i in range(n):
            loop_nodes = loops[i]
            for k in range(len(loop_nodes)):
                n1 = loop_nodes[k]
                n2 = loop_nodes[(k + 1) % len(loop_nodes)]
                if (n1, n2) == (u_edge, v_edge) or (n1, n2) == (v_edge, u_edge):
                    direction = 1 if (n1, n2) == (u_edge, v_edge) else -1
                    loops_containing_edge.append((i, direction))
                    break
        if len(loops_containing_edge) == 2:
            (i, dir_i), (j, dir_j) = loops_containing_edge
            Z[i, j] -= r_edge * dir_i * dir_j
            Z[j, i] -= r_edge * dir_i * dir_j

    return Z, graph, resistance_map

def calculate_potentials_v2(unknown_value, voltage_source_config, n, loops, Z_matrix, graph, resistance_map):
    """
    پتانسیل گره‌ها را با توجه به مقدار منبع مجهول محاسبه می‌کند.
    """
    vs = {}
    for (u, v), config in voltage_source_config.items():
        if config['type'] == 'fixed':
            vs[(u, v)] = config['value']
        elif config['type'] == 'e1_unknown':
            vs[(u, v)] = config['value'] * unknown_value
            
    V = np.zeros((n, 1), dtype=np.float64)
    for i, loop_nodes in enumerate(loops):
        loop_voltage = 0
        current_edges = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
        for u, v in current_edges:
            loop_voltage += vs.get((u, v), 0)
            loop_voltage -= vs.get((v, u), 0)
        V[i] = loop_voltage

    I = np.linalg.solve(Z_matrix, V)

    branch_currents = defaultdict(float)
    for i, loop_nodes in enumerate(loops):
        loop_current = I[i, 0]
        current_edges = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
        for u, v in current_edges:
            branch_currents[(u, v)] += loop_current

    all_nodes = sorted(graph.keys())
    if not all_nodes:
        return {}, branch_currents
    
    reference_node = max(all_nodes)
    potentials = {reference_node: 0.0}
    queue = deque([reference_node])
    visited = {reference_node}

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in visited:
                i_uv = branch_currents.get((u, v), 0.0) - branch_currents.get((v, u), 0.0)
                v_drop_r = i_uv * resistance_map.get((u, v), 0.0)
                v_gain_e = vs.get((u, v), 0.0) - vs.get((v, u), 0.0)
                potentials[v] = potentials[u] - v_drop_r + v_gain_e
                visited.add(v)
                queue.append(v)
                
    return potentials, branch_currents

def find_unknown_voltage(node_a, node_b, signed_voltage, **kwargs):
    """
    مقدار منبع مجهول را با حل معادله ولتاژ پیدا می‌کند.
    این تابع اکنون از یک روش بهینه‌سازی استفاده می‌کند.
    """
    def objective_function(unknown_value):
        potentials, _ = calculate_potentials_v2(unknown_value, **kwargs)
        if node_a not in potentials or node_b not in potentials:
            return float('inf')
        calculated_voltage = potentials[node_a] - potentials[node_b]
        return (calculated_voltage - signed_voltage)**2

    # از بهینه‌سازی برای پیدا کردن مقدار بهینه استفاده می‌شود
    result = minimize_scalar(objective_function)
    
    if not result.success:
        raise ValueError("Optimization failed. The selected node pair may not be sensitive to the unknown source.")
    
    return result.x

# --- بخش خروجی و PDF ---

def add_table_page(pdf, table_data, title, num_decimals, col_labels, format_text):
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.axis('off')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    t = ax.table(cellText=table_data,
                 colLabels=col_labels, loc='center', colWidths=[0.3, 0.3])
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1, 1.5)
    for (row, col), cell in t.get_celld().items():
        cell.set_text_props(ha='center', va='center')
        if row == 0: cell.set_facecolor('#4CAF50'); cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0: cell.set_facecolor('#f2f2f2')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def make_voltage_table_v2(potentials, num_decimals):
    all_nodes = sorted(potentials.keys())
    return [
        [f"({n1}, {n2})", f"{abs(potentials[n2] - potentials[n1]):.{num_decimals}f}"]
        for n1, n2 in itertools.combinations(all_nodes, 2)
    ]
    
def make_current_table_v2(branch_currents, num_decimals):
    return [
        [f"({u}, {v})", f"{(branch_currents.get((u, v), 0.0) * 1000):.{num_decimals}f}"]
        for u, v in sorted(branch_currents.keys())
    ]

def print_boxed_result(label, value, unit, num_decimals):
    text = f"{label}: {value:.{num_decimals}f} {unit}"
    width = len(text)
    print(f"┌{'─' * (width + 2)}┐")
    print(f"│ {text} │")
    print(f"└{'─' * (width + 2)}┘")

# --- تابع اصلی سازنده رابط کاربری ---

def create_analyzer_ui_v2(num_nodes, num_loops, voltage_source_config, edges, loops, num_decimals):
    Z_matrix, graph, resistance_map = precompute_system_v2(num_loops, loops, edges, voltage_source_config)
    all_nodes = sorted(graph.keys())
    
    # پیدا کردن منبع مجهول
    unknown_source_key = None
    for key, config in voltage_source_config.items():
        if config['type'] == 'e1_unknown':
            unknown_source_key = key
            break
    
    if unknown_source_key is None:
        raise ValueError("No unknown voltage source found with type 'e1_unknown'.")

    # لیست زوج گره‌ها برای ویجت کشویی
    all_node_pairs = [(f"({n1}, {n2})", (n1, n2)) for n1, n2 in itertools.combinations(all_nodes, 2)]

    # تعریف ویجت‌ها
    node_pair_dropdown = widgets.Dropdown(
        options=all_node_pairs,
        description='Node Pair:',
        layout=widgets.Layout(width='200px')
    )
    voltage_magnitude_box = widgets.FloatText(
        value=0.01,
        description='|Voltage| (V):',
        layout=widgets.Layout(width='200px')
    )
    voltage_sign_dropdown = widgets.Dropdown(
        options=[('V₁ > V₂ (Positive)', '+'), ('V₁ < V₂ (Negative)', '-'), ('Check both cases', 'both')],
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

    def on_calculate_clicked(b):
        with output_area:
            clear_output(wait=True)
            try:
                node_pair = node_pair_dropdown.value
                n1, n2 = node_pair
                v_magnitude = abs(voltage_magnitude_box.value)
                sign_choice = voltage_sign_dropdown.value

                calc_args = {
                    "voltage_source_config": voltage_source_config,
                    "n": num_loops,
                    "loops": loops,
                    "Z_matrix": Z_matrix,
                    "graph": graph,
                    "resistance_map": resistance_map
                }

                def process_case(signed_voltage, case_label):
                    print("\n" + "="*60 + f"\n{case_label}\n" + "="*60)
                    e1_result = find_unknown_voltage(n1, n2, signed_voltage, **calc_args)
                    print_boxed_result("Calculated value for unknown source", e1_result, "V", num_decimals)
                    potentials_result, branch_currents = calculate_potentials_v2(e1_result, **calc_args)
                    
                    voltage_table = make_voltage_table_v2(potentials_result, num_decimals)
                    current_table = make_current_table_v2(branch_currents, num_decimals)
                    
                    print("\nVoltage Table (V):")
                    print(tabulate(voltage_table, headers=["Nodes", "Voltage (V)"], tablefmt="fancy_grid", numalign="center"))
                    
                    print("\nBranch Current Table (mA):")
                    print(tabulate(current_table, headers=["Branch", "Current (mA)"], tablefmt="fancy_grid", numalign="center"))
                    
                    return voltage_table, current_table, e1_result

                if sign_choice == 'both':
                    pdf_filename = 'Results_both_cases.pdf'
                    with PdfPages(pdf_filename) as pdf:
                        voltage_table_pos, current_table_pos, e1_pos = process_case(v_magnitude, f"Case 1: V({n1}) > V({n2}), ΔV = +{v_magnitude:.{num_decimals}f} V")
                        add_table_page(pdf, voltage_table_pos, f"Voltage Table for Case 1 ($e_1 = {e1_pos:.{num_decimals}f}$ V)", num_decimals, ["Nodes", "Voltage (V)"], "voltage")
                        add_table_page(pdf, current_table_pos, f"Current Table for Case 1 ($e_1 = {e1_pos:.{num_decimals}f}$ V)", num_decimals, ["Branch", "Current (mA)"], "current")
                        
                        voltage_table_neg, current_table_neg, e1_neg = process_case(-v_magnitude, f"Case 2: V({n2}) > V({n1}), ΔV = -{v_magnitude:.{num_decimals}f} V")
                        add_table_page(pdf, voltage_table_neg, f"Voltage Table for Case 2 ($e_1 = {e1_neg:.{num_decimals}f}$ V)", num_decimals, ["Nodes", "Voltage (V)"], "voltage")
                        add_table_page(pdf, current_table_neg, f"Current Table for Case 2 ($e_1 = {e1_neg:.{num_decimals}f}$ V)", num_decimals, ["Branch", "Current (mA)"], "current")
                        
                    print(f"\nResults (including both cases) saved in '{pdf_filename}'.")

                else:
                    signed_voltage = v_magnitude if sign_choice == '+' else -v_magnitude
                    direction_text = f"V({n1}) > V({n2})" if sign_choice == '+' else f"V({n2}) > V({n1})"
                    voltage_table, current_table, e1_result = process_case(signed_voltage, f"Selected case: {direction_text}, ΔV = {signed_voltage:.{num_decimals}f} V")
                    
                    pdf_filename = 'Results.pdf'
                    with PdfPages(pdf_filename) as pdf:
                        add_table_page(pdf, voltage_table, f"Voltage Table ($e_1 = {e1_result:.{num_decimals}f}$ V)", num_decimals, ["Nodes", "Voltage (V)"], "voltage")
                        add_table_page(pdf, current_table, f"Current Table ($e_1 = {e1_result:.{num_decimals}f}$ V)", num_decimals, ["Branch", "Current (mA)"], "current")
                    print(f"\nTables saved to '{pdf_filename}'.")

            except Exception as e:
                print(f"❌ Error: {e}")

    # اتصال رویداد به دکمه
    calc_button.on_click(on_calculate_clicked)

    # چیدمان نهایی رابط کاربری
    ui_layout = widgets.VBox([
        widgets.HBox([node_pair_dropdown, voltage_magnitude_box, voltage_sign_dropdown]),
        calc_button,
        output_area
    ])
    
    return ui_layout