# Second part of the code (The Module) - FIXED VERSION
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate
from collections import defaultdict, deque
import itertools
import ipywidgets as widgets
from IPython.display import clear_output, HTML
import warnings
warnings.filterwarnings('ignore')

# --- Helper Functions (No changes needed here) ---
def find_reference_node(graph):
    return max(graph.keys())

def build_graph_from_edges(edges):
    graph = defaultdict(list)
    resistance_map = {}
    all_nodes = set()
    for u, v, r in edges:
        graph[u].append(v)
        graph[v].append(u)
        resistance_map[(u, v)] = resistance_map[(v, u)] = r
        all_nodes.add(u)
        all_nodes.add(v)
    return graph, resistance_map, sorted(all_nodes)

def precompute_system(n, loops, edges):
    Z = np.zeros((n, n), dtype=np.float64)
    # Diagonal
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
    # Off-diagonal
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
    graph, resistance_map, all_nodes = build_graph_from_edges(edges)
    ref_node = find_reference_node(graph)
    return Z, graph, resistance_map, all_nodes, ref_node

# FIXED: Modified function to handle the new unknown source format and fix current direction
def calculate_potentials(unknown_value, voltage_source_config, n, loops, Z_matrix,
                         graph, resistance_map, ref_node, unknown_source_info=None):
    vs = voltage_source_config.copy()
    if unknown_source_info:
        (u, v), sign = unknown_source_info
        vs[(u, v)] = unknown_value * sign

    V = np.zeros((n, 1), dtype=np.float64)
    for i, loop_nodes in enumerate(loops):
        loop_voltage = 0
        current_edges = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
        for u, v in current_edges:
            if not isinstance(vs.get((u, v)), tuple):
                loop_voltage += vs.get((u, v), 0)
            if not isinstance(vs.get((v, u)), tuple):
                loop_voltage -= vs.get((v, u), 0)
        V[i] = loop_voltage
    
    try:
        I = np.linalg.solve(Z_matrix, V)
    except np.linalg.LinAlgError:
        raise ValueError("Impedance matrix is singular. Check the circuit.")

    # FIXED: Proper calculation of branch currents
    branch_currents = {}
    
    # Initialize all possible branch currents to zero
    for u, v, r in edges:
        branch_currents[(u, v)] = 0.0
        branch_currents[(v, u)] = 0.0
    
    # Add contributions from each loop
    for i, loop_nodes in enumerate(loops):
        loop_current = I[i, 0]
        current_edges = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
        
        for u, v in current_edges:
            # Loop current flows in the direction defined by the loop
            branch_currents[(u, v)] += loop_current

    potentials = {ref_node: 0.0}
    queue = deque([ref_node])
    visited = {ref_node}
    
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in visited:
                # Net current from u to v
                i_uv = branch_currents.get((u, v), 0.0) - branch_currents.get((v, u), 0.0)
                v_drop_r = i_uv * resistance_map.get((u, v), 0.0)
                
                v_gain_e_uv = vs.get((u, v), 0.0) if not isinstance(vs.get((u,v)), tuple) else 0.0
                v_gain_e_vu = vs.get((v, u), 0.0) if not isinstance(vs.get((v,u)), tuple) else 0.0
                v_gain_e = v_gain_e_uv - v_gain_e_vu
                
                potentials[v] = potentials[u] - v_drop_r + v_gain_e
                visited.add(v)
                queue.append(v)
    
    return potentials, branch_currents

def find_unknown_voltage(node_a, node_b, target_voltage, voltage_source_config,
                         n, loops, Z_matrix, graph, resistance_map, ref_node,
                         unknown_source_info=None, decimal_places=4):
    
    pot0, _ = calculate_potentials(0.0, voltage_source_config, n, loops, Z_matrix,
                                   graph, resistance_map, ref_node, unknown_source_info)
    pot1, _ = calculate_potentials(1.0, voltage_source_config, n, loops, Z_matrix,
                                   graph, resistance_map, ref_node, unknown_source_info)

    v0 = pot0[node_a] - pot0[node_b]
    v1 = pot1[node_a] - pot1[node_b]
    
    k = v1 - v0
    
    if abs(k) < 1e-12:
        raise ValueError("This node pair is not sensitive to the unknown source! Choose different nodes.")
        
    unknown_value = (target_voltage - v0) / k
    return round(unknown_value, decimal_places)

def make_voltage_table(potentials, all_nodes, decimal_places=4):
    return [
        [f"({n1}, {n2})", f"{abs(potentials[n2] - potentials[n1]):.{decimal_places}f}"]
        for n1, n2 in itertools.combinations(all_nodes, 2)
    ]

# FIXED: Corrected current table generation
def make_current_table(branch_currents, edges, decimal_places=4):
    table = []
    for (u, v, r) in edges:
        # Get the net current from u to v
        i_net = branch_currents.get((u, v), 0.0) - branch_currents.get((v, u), 0.0)
        current_mA = i_net * 1000
        
        # Remove numerical noise
        if abs(current_mA) < 1e-10:
            current_mA = 0.0
        
        # Determine direction and magnitude
        if current_mA >= 0:
            direction = f"{u} → {v}"
            magnitude = current_mA
        else:
            direction = f"{v} → {u}"
            magnitude = abs(current_mA)
        
        table.append([f"({u}, {v})", f"{magnitude:.{decimal_places}f}", direction])
    
    return table

def make_combined_table(voltage_table, current_table, decimal_places=4):
    combined = []
    current_map = {row[0]: (row[1], row[2]) for row in current_table}
    for node_pair, v in voltage_table:
        if node_pair in current_map:
            i, direction = current_map[node_pair]
        else:
            i, direction = "–", "–"
        combined.append([node_pair, v, i, direction])
    return combined

def add_combined_table_page(pdf, table_data, unknown_value, unknown_name, decimal_places):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    ax.set_title(f"Voltage and Current Table\nfor {unknown_name} = {unknown_value:.{decimal_places}f} V",
                 fontsize=14, weight='bold', pad=20, fontfamily='DejaVu Sans', loc='center')
    t = ax.table(cellText=table_data,
                 colLabels=["Nodes", "Voltage (V)", "Current (mA)", "Direction"],
                 loc='center', colWidths=[0.15, 0.15, 0.15, 0.15])
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1.3, 1.3)
    for (row, col), cell in t.get_celld().items():
        cell.set_text_props(ha='center', va='center')
        if row == 0:
            cell.set_facecolor('#673AB7')
            cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0:
            cell.set_facecolor('#f3e5f5')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def print_boxed_value(value, name, decimal_places):
    text = f"{name} = {value:.{decimal_places}f} V"
    width = len(text)
    print(f"┌{'─' * (width + 2)}┐")
    print(f"│ {text} │")
    print(f"└{'─' * (width + 2)}┘")

def create_analyzer_ui(n, voltage_source_config, edges, loops, decimal_places=4, num_nodes=8):
    Z_matrix, graph, resistance_map, all_nodes, ref_node = precompute_system(n, loops, edges)
    
    unknown_source_info = None
    for (u, v), val in voltage_source_config.items():
        if isinstance(val, tuple) and val[0] == 'unknown':
            sign = val[1]
            if sign not in [1, -1]:
                raise ValueError(f"Sign for unknown source at ({u},{v}) must be 1 or -1.")
            unknown_source_info = ((u, v), sign)
            break
    
    if not unknown_source_info:
        raise ValueError("No unknown source defined! Use ('unknown', 1) or ('unknown', -1).")

    unknown_source_nodes, _ = unknown_source_info
    unknown_name = f"e({unknown_source_nodes[0]},{unknown_source_nodes[1]})"

    node_pairs = [f"({n1}, {n2})" for n1, n2 in itertools.combinations(all_nodes, 2)]
    node_pair_dropdown = widgets.Dropdown(options=node_pairs, value=node_pairs[0], description='Node Pair:', layout=widgets.Layout(width='300px'))
    voltage_magnitude_box = widgets.FloatText(value=0.01, description='Voltage (V):', layout=widgets.Layout(width='200px'))
    voltage_sign_dropdown = widgets.Dropdown(options=[('V₁ > V₂ (Positive)', '+'), ('V₂ > V₁ (Negative)', '-'), ('Check Both Cases', 'both')], value='both', description='Direction:', layout=widgets.Layout(width='250px'))
    calc_button = widgets.Button(description="Calculate", button_style='success', layout=widgets.Layout(width='150px'))
    output_area = widgets.Output()

    def on_calculate_clicked(b):
        with output_area:
            clear_output(wait=True)
            try:
                selected_pair = node_pair_dropdown.value
                n1, n2 = map(int, selected_pair.strip('()').split(', '))
                v_magnitude = abs(voltage_magnitude_box.value)
                sign_choice = voltage_sign_dropdown.value
                
                calc_args = {
                    "voltage_source_config": voltage_source_config, "n": n, "loops": loops,
                    "Z_matrix": Z_matrix, "graph": graph, "resistance_map": resistance_map,
                    "ref_node": ref_node, "unknown_source_info": unknown_source_info
                }
                
                find_args = calc_args.copy()
                find_args["decimal_places"] = decimal_places

                if sign_choice == 'both':
                    print("="*70); print("Checking both possible cases:"); print("="*70)
                    pdf_filename = 'Circuit_Analysis_Results.pdf'
                    with PdfPages(pdf_filename) as pdf:
                        # Case 1
                        print(f"\nCase 1: V({n1}) > V({n2}), ΔV = +{v_magnitude:.{decimal_places}f} V")
                        e_pos = find_unknown_voltage(n1, n2, v_magnitude, **find_args)
                        potentials_pos, currents_pos = calculate_potentials(e_pos, **calc_args)
                        voltage_table_pos = make_voltage_table(potentials_pos, all_nodes, decimal_places)
                        current_table_pos = make_current_table(currents_pos, edges, decimal_places)
                        combined_pos = make_combined_table(voltage_table_pos, current_table_pos, decimal_places)
                        print_boxed_value(e_pos, unknown_name, decimal_places)
                        print(tabulate(combined_pos, headers=["Nodes", "Voltage (V)", "Current (mA)", "Direction"], tablefmt="fancy_grid", numalign="center"))
                        add_combined_table_page(pdf, combined_pos, e_pos, unknown_name, decimal_places)

                        # Case 2
                        print(f"\n{'-'*50}"); print(f"Case 2: V({n2}) > V({n1}), ΔV = -{v_magnitude:.{decimal_places}f} V")
                        e_neg = find_unknown_voltage(n1, n2, -v_magnitude, **find_args)
                        potentials_neg, currents_neg = calculate_potentials(e_neg, **calc_args)
                        voltage_table_neg = make_voltage_table(potentials_neg, all_nodes, decimal_places)
                        current_table_neg = make_current_table(currents_neg, edges, decimal_places)
                        combined_neg = make_combined_table(voltage_table_neg, current_table_neg, decimal_places)
                        print_boxed_value(e_neg, unknown_name, decimal_places)
                        print(tabulate(combined_neg, headers=["Nodes", "Voltage (V)", "Current (mA)", "Direction"], tablefmt="fancy_grid", numalign="center"))
                        if abs(e_pos - e_neg) > 1e-9:
                            add_combined_table_page(pdf, combined_neg, e_neg, unknown_name, decimal_places)
                    print(f"\nResults saved in '{pdf_filename}'.")
                else:
                    signed_voltage = v_magnitude if sign_choice == '+' else -v_magnitude
                    direction_text = f"V({n1}) > V({n2})" if sign_choice == '+' else f"V({n2}) > V({n1})"
                    print(f"\nSelected Case: {direction_text}, ΔV = {signed_voltage:.{decimal_places}f} V")
                    e_result = find_unknown_voltage(n1, n2, signed_voltage, **find_args)
                    potentials_result, currents_result = calculate_potentials(e_result, **calc_args)
                    voltage_table = make_voltage_table(potentials_result, all_nodes, decimal_places)
                    current_table = make_current_table(currents_result, edges, decimal_places)
                    combined_table = make_combined_table(voltage_table, current_table, decimal_places)
                    print_boxed_value(e_result, unknown_name, decimal_places)
                    print(tabulate(combined_table, headers=["Nodes", "Voltage (V)", "Current (mA)", "Direction"], tablefmt="fancy_grid", numalign="center"))
                    with PdfPages('Circuit_Analysis_Results.pdf') as pdf:
                        add_combined_table_page(pdf, combined_table, e_result, unknown_name, decimal_places)
                    print("\nResults saved in 'Circuit_Analysis_Results.pdf'.")
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

    calc_button.on_click(on_calculate_clicked)
    ui_layout = widgets.VBox([
        widgets.HTML(f"<h3>Inverse Circuit Analysis - Unknown Source: {unknown_name}</h3>"),
        widgets.HTML(f"<p>Reference Node: {ref_node}</p>"),
        node_pair_dropdown,
        widgets.HBox([voltage_magnitude_box, voltage_sign_dropdown]),
        calc_button,
        output_area
    ])
    return ui_layout