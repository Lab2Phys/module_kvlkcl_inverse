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
# --- Helper Functions ---
def find_reference_node(graph):
    """Find the reference node automatically (highest number)"""
    return max(graph.keys())
def build_graph_from_edges(edges):
    """Build a graph from the list of edges"""
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
def precompute_system(n, loops, edges, unknown_source=None):
    """Precompute the impedance matrix with support for any unknown source"""
    Z = np.zeros((n, n), dtype=np.float64)
   
    # Calculate main diagonal elements
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
   
    # Calculate off-diagonal elements
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
def calculate_potentials(unknown_value, voltage_source_config, n, loops, Z_matrix,
                        graph, resistance_map, ref_node, unknown_source=None):
    """Calculate node potentials"""
    vs = voltage_source_config.copy()
   
    # Replace the unknown value
    if unknown_source:
        for key, val in vs.items():
            if val == 'unknown': # Changed from -1 to 'unknown'
                vs[key] = unknown_value
   
    V = np.zeros((n, 1), dtype=np.float64)
   
    # Calculate voltage vector
    for i, loop_nodes in enumerate(loops):
        loop_voltage = 0
        current_edges = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
        for u, v in current_edges:
            loop_voltage += vs.get((u, v), 0)
            loop_voltage -= vs.get((v, u), 0)
        V[i] = loop_voltage
   
    # Solve the system of equations
    try:
        I = np.linalg.solve(Z_matrix, V)
    except np.linalg.LinAlgError:
        raise ValueError("Impedance matrix is singular. Check the circuit.")
   
    # Calculate branch currents
    branch_currents = defaultdict(float)
    for i, loop_nodes in enumerate(loops):
        loop_current = I[i, 0]
        current_edges = list(zip(loop_nodes, loop_nodes[1:] + loop_nodes[:1]))
        for u, v in current_edges:
            branch_currents[(u, v)] += loop_current
   
    # Calculate node potentials using BFS
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
def find_unknown_voltage(node_a, node_b, target_voltage, voltage_source_config,
                        n, loops, Z_matrix, graph, resistance_map, ref_node,
                        unknown_source=None, decimal_places=4):
    """Find the unknown voltage source"""
    # Calculate for two test values
    pot0, _ = calculate_potentials(0.0, voltage_source_config, n, loops, Z_matrix,
                                 graph, resistance_map, ref_node, unknown_source)
    pot1, _ = calculate_potentials(1.0, voltage_source_config, n, loops, Z_matrix,
                                 graph, resistance_map, ref_node, unknown_source)
   
    v0 = pot0[node_a] - pot0[node_b]
    v1 = pot1[node_a] - pot1[node_b]
   
    k = v1 - v0
    if abs(k) < 1e-12:
        raise ValueError("This node pair is not sensitive to the unknown source! Choose different nodes.")
   
    unknown_value = (target_voltage - v0) / k
    return round(unknown_value, decimal_places)
def make_voltage_table(potentials, all_nodes, decimal_places=4):
    """Create a table of voltages between nodes"""
    return [
        [f"({n1}, {n2})", f"{abs(potentials[n2] - potentials[n1]):.{decimal_places}f}"]
        for n1, n2 in itertools.combinations(all_nodes, 2)
    ]
def make_current_table(branch_currents, edges, decimal_places=4):
    """Create a table of branch currents"""
    table = []
    for (u, v, r) in edges:
        i_uv = branch_currents.get((u, v), 0.0) - branch_currents.get((v, u), 0.0)
        current_mA = i_uv * 1000 # Convert to milliamperes
       
        if abs(current_mA) < 1e-10: # Remove very small currents
            current_mA = 0.0
       
        direction = f"{u} → {v}" if current_mA >= 0 else f"{v} → {u}"
        table.append([f"({u}, {v})", f"{abs(current_mA):.{decimal_places}f}", direction, f"{r}"])
   
    return table
# --- Output and PDF Functions ---
def add_voltage_table_page(pdf, table_data, unknown_value, unknown_name, decimal_places):
    """Add voltage table page to PDF"""
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.axis('off')
    ax.set_title(f"Voltage Table Between Nodes - {unknown_name} = {unknown_value:.{decimal_places}f} V",
                fontsize=14, weight='bold', pad=20, fontfamily='DejaVu Sans')
   
    t = ax.table(cellText=table_data, colLabels=["Nodes", "Voltage (V)"],
                loc='center', colWidths=[0.2, 0.2])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.2)
   
    for (row, col), cell in t.get_celld().items():
        cell.set_text_props(ha='center', va='center')
        if row == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')
   
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
def add_current_table_page(pdf, table_data, unknown_value, unknown_name, decimal_places):
    """Add current table page to PDF"""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.axis('off')
    ax.set_title(f"Branch Current Table - {unknown_name} = {unknown_value:.{decimal_places}f} V",
                fontsize=14, weight='bold', pad=20, fontfamily='DejaVu Sans')
   
    t = ax.table(cellText=table_data, colLabels=["Branch", "Current (mA)", "Direction"],
                loc='center', colWidths=[0.15, 0.15, 0.15])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.2)
   
    for (row, col), cell in t.get_celld().items():
        cell.set_text_props(ha='center', va='center')
        if row == 0:
            cell.set_facecolor('#2196F3')
            cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0:
            cell.set_facecolor('#e3f2fd')
   
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
def print_boxed_value(value, name, decimal_places):
    """Print value in a boxed format"""
    text = f"{name} = {value:.{decimal_places}f} V"
    width = len(text)
    print(f"┌{'─' * (width + 2)}┐")
    print(f"│ {text} │")
    print(f"└{'─' * (width + 2)}┘")
# --- Main UI Function ---
def create_analyzer_ui(n, voltage_source_config, edges, loops, decimal_places=4, num_nodes=8):
    """Create a complete UI for inverse circuit analysis"""
   
    # Precomputations
    Z_matrix, graph, resistance_map, all_nodes, ref_node = precompute_system(n, loops, edges)
   
    # Identify unknown source
    unknown_source = None
    for (u, v), val in voltage_source_config.items():
        if val == 'unknown': # Changed from -1 to 'unknown'
            unknown_source = (u, v)
            break
   
    if not unknown_source:
        raise ValueError("No unknown source defined!")
   
    unknown_name = f"e{unknown_source[0]}{unknown_source[1]}"
   
    # Create list of node pairs for Dropdown
    node_pairs = [f"({n1}, {n2})" for n1, n2 in itertools.combinations(all_nodes, 2)]
   
    # Widgets
    node_pair_dropdown = widgets.Dropdown(
        options=node_pairs,
        value=node_pairs[0],
        description='Node Pair:',
        layout=widgets.Layout(width='300px')
    )
   
    voltage_magnitude_box = widgets.FloatText(
        value=0.01,
        description='Voltage (V):',
        layout=widgets.Layout(width='200px')
    )
   
    voltage_sign_dropdown = widgets.Dropdown(
        options=[('V₁ > V₂ (Positive)', '+'), ('V₂ > V₁ (Negative)', '-'), ('Check Both Cases', 'both')],
        value='both',
        description='Direction:',
        layout=widgets.Layout(width='250px')
    )
   
    calc_button = widgets.Button(
        description="Calculate",
        button_style='success',
        layout=widgets.Layout(width='150px')
    )
   
    output_area = widgets.Output()
   
    # Event handler
    def on_calculate_clicked(b):
        with output_area:
            clear_output(wait=True)
            try:
                # Extract nodes from Dropdown
                selected_pair = node_pair_dropdown.value
                n1, n2 = map(int, selected_pair.strip('()').split(', '))
               
                v_magnitude = abs(voltage_magnitude_box.value)
                sign_choice = voltage_sign_dropdown.value
               
                # Arguments needed for calculate_potentials
                calc_args = {
                    "voltage_source_config": voltage_source_config,
                    "n": n,
                    "loops": loops,
                    "Z_matrix": Z_matrix,
                    "graph": graph,
                    "resistance_map": resistance_map,
                    "ref_node": ref_node,
                    "unknown_source": unknown_source
                }
               
                # Arguments needed for find_unknown_voltage
                find_args = calc_args.copy()
                find_args["decimal_places"] = decimal_places
               
                if sign_choice == 'both':
                    print("="*70)
                    print("Checking both possible cases:")
                    print("="*70)
                   
                    pdf_filename = 'Circuit_Analysis_Results.pdf'
                    with PdfPages(pdf_filename) as pdf:
                        # Case 1: V(n1) > V(n2)
                        print(f"\nCase 1: V({n1}) > V({n2}), ΔV = +{v_magnitude:.{decimal_places}f} V")
                        e_pos = find_unknown_voltage(n1, n2, v_magnitude, **find_args)
                        potentials_pos, currents_pos = calculate_potentials(e_pos, **calc_args)
                       
                        voltage_table_pos = make_voltage_table(potentials_pos, all_nodes, decimal_places)
                        current_table_pos = make_current_table(currents_pos, edges, decimal_places)
                       
                        print_boxed_value(e_pos, unknown_name, decimal_places)
                        print("\nVoltage Table (Positive Case):")
                        print(tabulate(voltage_table_pos, headers=["Nodes", "Voltage (V)"],
                                      tablefmt="fancy_grid", numalign="center"))
                       
                        add_voltage_table_page(pdf, voltage_table_pos, e_pos, unknown_name, decimal_places)
                        add_current_table_page(pdf, current_table_pos, e_pos, unknown_name, decimal_places)
                       
                        # Case 2: V(n2) > V(n1)
                        print(f"\n{'-'*50}")
                        print(f"Case 2: V({n2}) > V({n1}), ΔV = -{v_magnitude:.{decimal_places}f} V")
                        e_neg = find_unknown_voltage(n1, n2, -v_magnitude, **find_args)
                        potentials_neg, currents_neg = calculate_potentials(e_neg, **calc_args)
                       
                        voltage_table_neg = make_voltage_table(potentials_neg, all_nodes, decimal_places)
                        current_table_neg = make_current_table(currents_neg, edges, decimal_places)
                       
                        print_boxed_value(e_neg, unknown_name, decimal_places)
                        print("\nVoltage Table (Negative Case):")
                        print(tabulate(voltage_table_neg, headers=["Nodes", "Voltage (V)"],
                                      tablefmt="fancy_grid", numalign="center"))
                       
                        if abs(e_pos - e_neg) > 1e-9:
                            add_voltage_table_page(pdf, voltage_table_neg, e_neg, unknown_name, decimal_places)
                            add_current_table_page(pdf, current_table_neg, e_neg, unknown_name, decimal_places)
                   
                    print(f"\nResults saved in '{pdf_filename}'.")
                   
                else:
                    signed_voltage = v_magnitude if sign_choice == '+' else -v_magnitude
                    direction_text = f"V({n1}) > V({n2})" if sign_choice == '+' else f"V({n2}) > V({n1})"
                   
                    print(f"\nSelected Case: {direction_text}, ΔV = {signed_voltage:.{decimal_places}f} V")
                    e_result = find_unknown_voltage(n1, n2, signed_voltage, **find_args)
                    potentials_result, currents_result = calculate_potentials(e_result, **calc_args)
                   
                    voltage_table = make_voltage_table(potentials_result, all_nodes, decimal_places)
                    current_table = make_current_table(currents_result, edges, decimal_places)
                   
                    print_boxed_value(e_result, unknown_name, decimal_places)
                   
                    print("\nVoltage Table:")
                    print(tabulate(voltage_table, headers=["Nodes", "Voltage (V)"],
                                  tablefmt="fancy_grid", numalign="center"))
                   
                    print("\nCurrent Table:")
                    print(tabulate(current_table, headers=["Branch", "Current (mA)", "Direction"],
                                  tablefmt="fancy_grid", numalign="center"))
                   
                    with PdfPages('Circuit_Analysis_Results.pdf') as pdf:
                        add_voltage_table_page(pdf, voltage_table, e_result, unknown_name, decimal_places)
                        add_current_table_page(pdf, current_table, e_result, unknown_name, decimal_places)
                   
                    print("\nResults saved in 'Circuit_Analysis_Results.pdf'.")
                   
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
   
    calc_button.on_click(on_calculate_clicked)
   
    # UI Layout
    ui_layout = widgets.VBox([
        widgets.HTML(f"<h3>Inverse Circuit Analysis - Unknown Source: {unknown_name}</h3>"),
        widgets.HTML(f"<p>Reference Node: {ref_node}</p>"),
        node_pair_dropdown,
        widgets.HBox([voltage_magnitude_box, voltage_sign_dropdown]),
        calc_button,
        output_area
    ])
   
    return ui_layout