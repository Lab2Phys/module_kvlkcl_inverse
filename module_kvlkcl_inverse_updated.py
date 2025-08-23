import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate
from collections import defaultdict, deque
import itertools
import ipywidgets as widgets
from IPython.display import clear_output, display

# --- Calculation Functions ---
def precompute_system(n, loops, edges):
    num_nodes = len(set(u for u, _, _ in edges) | set(v for _, v, _ in edges))
    num_edges = len(edges)
    expected_loops = num_edges - num_nodes + 1
    if n != expected_loops:
        raise ValueError(f"Expected {expected_loops} loops, but got {n}")
    Z = np.zeros((n, n), dtype=np.float64)
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
            Z[i, j] += r_edge * dir_i * dir_j
            Z[j, i] += r_edge * dir_i * dir_j
    graph = defaultdict(list)
    resistance_map = {}
    for u, v, r in edges:
        graph[u].append(v)
        graph[v].append(u)
        resistance_map[(u, v)] = resistance_map[(v, u)] = r
    return Z, graph, resistance_map

def calculate_potentials(e1_value, voltage_source_config, n, loops, Z_matrix, graph, resistance_map, ref_node):
    vs = {}
    for (u, v), val in voltage_source_config.items():
        vs[(u, v)] = val * e1_value if val == -1 else val
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
    potentials = {ref_node: 0.0}
    queue = deque([ref_node])
    visited = {ref_node}
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

def find_unknown_voltage(node_a, node_b, signed_voltage, unknown_edge, **kwargs):
    voltage_source_config = kwargs["voltage_source_config"]
    kwargs_no_edge = {k: v for k, v in kwargs.items() if k != "unknown_edge"}
    pot0 = calculate_potentials(0.0, voltage_source_config={k: v for k, v in voltage_source_config.items()}, **kwargs_no_edge)
    pot1 = calculate_potentials(1.0, voltage_source_config={k: (v if k != unknown_edge else -1) for k, v in voltage_source_config.items()}, **kwargs_no_edge)
    v0 = pot0[node_a] - pot0[node_b]
    v1 = pot1[node_a] - pot1[node_b]
    k = v1 - v0
    if abs(k) < 1e-8:
        raise ValueError(f"This node pair is not sensitive to the voltage source at {unknown_edge}! Try different nodes.")
    return (signed_voltage - v0) / k

def make_voltage_table(potentials, all_nodes, decimal_places=4):
    return [
        [f"({n1}, {n2})", f"{abs(potentials[n2] - potentials[n1]):.{decimal_places}f}"]
        for n1, n2 in itertools.combinations(all_nodes, 2)
    ]

def make_current_table(branch_currents, edges, decimal_places=4):
    table = []
    for u, v, _ in edges:
        current = branch_currents.get((u, v), 0.0) - branch_currents.get((v, u), 0.0)
        current_mA = current * 1000
        table.append([f"({u}, {v})", f"{current_mA:.{decimal_places}f}"])
    return table

# --- Output and PDF Functions ---
def add_table_page(pdf, table_data, e1_value, decimal_places=4):
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.axis('off')
    ax.set_title(f"Voltage table for calculated $e_1 = {e1_value:.{decimal_places}f}$ V", fontsize=14, weight='bold', pad=20)
    t = ax.table(cellText=table_data, colLabels=["Nodes", "Voltage (V)"], loc='center', colWidths=[0.3, 0.3])
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 1.5)
    for (row, col), cell in t.get_celld().items():
        cell.set_text_props(ha='center', va='center')
        if row == 0: cell.set_facecolor('#4CAF50'); cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0: cell.set_facecolor('#f2f2f2')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def add_current_table_page(pdf, table_data, e1_value, decimal_places=4):
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.axis('off')
    ax.set_title(f"Current table for calculated $e_1 = {e1_value:.{decimal_places}f}$ V", fontsize=14, weight='bold', pad=20)
    t = ax.table(cellText=table_data, colLabels=["Edges", "Current (mA)"], loc='center', colWidths=[0.3, 0.3])
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 1.5)
    for (row, col), cell in t.get_celld().items():
        cell.set_text_props(ha='center', va='center')
        if row == 0: cell.set_facecolor('#4CAF50'); cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0: cell.set_facecolor('#f2f2f2')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def print_boxed_e1(e1_value, decimal_places=4):
    text = f"Calculated value: e1 = {e1_value:.{decimal_places}f} V"
    width = len(text)
    print(f"┌{'─' * (width + 2)}┐")
    print(f"│ {text} │")
    print(f"└{'─' * (width + 2)}┘")

# --- Main UI Function ---
def create_analyzer_ui(n, voltage_source_config, edges, loops, decimal_places=4, unknown_edge=(1, 3)):
    """
    Creates and displays the UI for circuit analysis.
    """
    # Set reference node to the highest node number
    ref_node = max(max(u, v) for u, v, _ in edges)
    Z_matrix, graph, resistance_map = precompute_system(n, loops, edges)
    all_nodes = sorted(graph.keys())
    node_pairs = [(n1, n2) for n1, n2 in itertools.combinations(all_nodes, 2)]
    node_pair_options = [(f"({n1}, {n2})", (n1, n2)) for n1, n2 in node_pairs]
    
    # Define analysis widgets
    node_pair_dropdown = widgets.Dropdown(
        options=node_pair_options,
        description='Node Pair:',
        layout=widgets.Layout(width='300px')
    )
    voltage_magnitude_box = widgets.FloatText(value=0.01, description='|Voltage| (V):', layout=widgets.Layout(width='200px'))
    voltage_sign_dropdown = widgets.Dropdown(
        options=[('V₁ > V₂ (Positive)', '+'), ('V₁ < V₂ (Negative)', '-'), ('Check both cases', 'both')],
        value='both', description='Direction:', layout=widgets.Layout(width='300px'))
    calc_button = widgets.Button(description="Calculate e1", button_style='success', layout=widgets.Layout(width='150px'))
    output_area = widgets.Output()

    def on_calculate_clicked(b):
        with output_area:
            clear_output(wait=True)
            try:
                n1, n2 = node_pair_dropdown.value
                v_magnitude, sign_choice = abs(voltage_magnitude_box.value), voltage_sign_dropdown.value
                if n1 not in all_nodes or n2 not in all_nodes or n1 == n2:
                    print("❌ Error: Invalid node pair.")
                    return

                calc_args = {
                    "voltage_source_config": voltage_source_config,
                    "n": n,
                    "loops": loops,
                    "Z_matrix": Z_matrix,
                    "graph": graph,
                    "resistance_map": resistance_map,
                    "ref_node": ref_node,
                    "unknown_edge": unknown_edge
                }

                if sign_choice == 'both':
                    print("\n" + "="*60 + "\nChecking both possible cases:\n" + "="*60)
                    pdf_filename = 'Exp_voltages.pdf'
                    current_pdf_filename = 'Exp_currents.pdf'
                    with PdfPages(pdf_filename) as pdf, PdfPages(current_pdf_filename) as current_pdf:
                        print(f"\nCase 1: V({n1}) > V({n2}), ΔV = +{v_magnitude:.{decimal_places}f} V")
                        e1_pos = find_unknown_voltage(n1, n2, v_magnitude, unknown_edge, **calc_args)
                        potentials_pos, currents_pos = calculate_potentials(e1_pos, **calc_args)
                        table_pos = make_voltage_table(potentials_pos, all_nodes, decimal_places)
                        current_table_pos = make_current_table(currents_pos, edges, decimal_places)
                        print_boxed_e1(e1_pos, decimal_places)
                        print(tabulate(table_pos, headers=["Nodes", "Voltage (V)"], tablefmt="fancy_grid", numalign="center"))
                        print("\nCurrents (mA):")
                        print(tabulate(current_table_pos, headers=["Edges", "Current (mA)"], tablefmt="fancy_grid", numalign="center"))

                        print(f"\nCase 2: V({n2}) > V({n1}), ΔV = -{v_magnitude:.{decimal_places}f} V")
                        e1_neg = find_unknown_voltage(n1, n2, -v_magnitude, unknown_edge, **calc_args)
                        potentials_neg, currents_neg = calculate_potentials(e1_neg, **calc_args)
                        table_neg = make_voltage_table(potentials_neg, all_nodes, decimal_places)
                        current_table_neg = make_current_table(currents_neg, edges, decimal_places)
                        print_boxed_e1(e1_neg, decimal_places)
                        print(tabulate(table_neg, headers=["Nodes", "Voltage (V)"], tablefmt="fancy_grid", numalign="center"))
                        print("\nCurrents (mA):")
                        print(tabulate(current_table_neg, headers=["Edges", "Current (mA)"], tablefmt="fancy_grid", numalign="center"))

                        add_table_page(pdf, table_pos, e1_pos, decimal_places)
                        add_current_table_page(current_pdf, current_table_pos, e1_pos, decimal_places)
                        if abs(e1_pos - e1_neg) > 1e-9:
                            add_table_page(pdf, table_neg, e1_neg, decimal_places)
                            add_current_table_page(current_pdf, current_table_neg, e1_neg, decimal_places)
                    print(f"\nVoltage tables saved in '{pdf_filename}'.")
                    print(f"Current tables saved in '{current_pdf_filename}'.")
                else:
                    signed_voltage = v_magnitude if sign_choice == '+' else -v_magnitude
                    direction_text = f"V({n1}) > V({n2})" if sign_choice == '+' else f"V({n2}) > V({n1})"
                    print(f"\nSelected case: {direction_text}, ΔV = {signed_voltage:.{decimal_places}f} V")
                    e1_result = find_unknown_voltage(n1, n2, signed_voltage, unknown_edge, **calc_args)
                    potentials_result, currents_result = calculate_potentials(e1_result, **calc_args)
                    table_result = make_voltage_table(potentials_result, all_nodes, decimal_places)
                    current_table_result = make_current_table(currents_result, edges, decimal_places)
                    print_boxed_e1(e1_result, decimal_places)
                    print(tabulate(table_result, headers=["Nodes", "Voltage (V)"], tablefmt="fancy_grid", numalign="center"))
                    print("\nCurrents (mA):")
                    print(tabulate(current_table_result, headers=["Edges", "Current (mA)"], tablefmt="fancy_grid", numalign="center"))
                    with PdfPages('Exp_voltages.pdf') as pdf, PdfPages('Exp_currents.pdf') as current_pdf:
                        add_table_page(pdf, table_result, e1_result, decimal_places)
                        add_current_table_page(current_pdf, current_table_result, e1_result, decimal_places)
                    print("\nVoltage table saved to 'Exp_voltages.pdf'.")
                    print("\nCurrent table saved to 'Exp_currents.pdf'.")
            except Exception as e:
                print(f"❌ Error: {e}")

    calc_button.on_click(on_calculate_clicked)
    ui_layout = widgets.VBox([
        node_pair_dropdown,
        widgets.HBox([voltage_magnitude_box, voltage_sign_dropdown]),
        calc_button,
        output_area
    ])
    display(ui_layout)