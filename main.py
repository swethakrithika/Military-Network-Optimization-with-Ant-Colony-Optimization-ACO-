import streamlit as st
import time
import matplotlib.pyplot as plt
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import heapq
from collections import defaultdict, deque
import math

# ================== Constants ==================
MAX_CAPACITY_UTILIZATION = 0.8  # 80% utilization threshold
MIN_RELIABILITY = 0.7  # Minimum acceptable link reliability
MAX_LATENCY = 300  # ms maximum allowed latency
MIN_BANDWIDTH = {
    "command": 2.0,  # Mbps
    "video": 5.0,
    "sensor": 1.0,
    "data": 0.5
}
ACO_ITERATIONS = 50
NUM_ANTS = 20
EVAPORATION_RATE = 0.3
PHEROMONE_DEPOSIT = 1.0
ALPHA = 1.0  # Pheromone importance
BETA = 2.0   # Heuristic importance

# ================== Enhanced Network Models ==================
@dataclass
class NetworkNode:
    id: str
    position: Tuple[float, float, float]
    status: str = "active"  # active/jammed/destroyed
    comm_range: float = 100.0
    bandwidth: float = 10.0  # Mbps capacity
    current_load: float = 0.0  # Current traffic load
    queue_delay: float = 0.0  # Processing delay
    buffer_size: int = 100  # Packet buffer size
    buffer_occupancy: int = 0  # Current buffer usage

@dataclass
class NetworkLink:
    source: str
    target: str
    latency: float  # ms
    max_bandwidth: float  # Mbps
    current_utilization: float = 0.0
    reliability: float = 0.95  # Packet success rate
    pheromone: float = 0.1  # For ACO routing

@dataclass
class NetworkFlow:
    source: str
    destination: str
    traffic_class: str
    required_bandwidth: float
    max_latency: float
    route: List[str] = None
    active: bool = False

class MilitaryNetwork:
    def __init__(self, num_nodes=15):
        self.nodes = self._create_nodes(num_nodes)
        self.links = self._initialize_links()
        self.jamming_zones = []
        self.flows = []
        self.priority_classes = {
            "command": 0,  # Highest priority
            "video": 1,
            "sensor": 2,
            "data": 3  # Lowest priority
        }
        self.pheromone_matrix = defaultdict(dict)
        self._initialize_pheromones()
    
    def _create_nodes(self, num_nodes):
        nodes = [NetworkNode("Command", (0, 0, 0), comm_range=200, bandwidth=50.0, buffer_size=500)]
        for i in range(1, num_nodes):
            node_type = random.choice(["Drone", "Soldier", "Vehicle"])
            x, y = random.uniform(-200, 200), random.uniform(-200, 200)
            z = random.uniform(0, 50) if node_type == "Drone" else 0
            bandwidth = random.choice([5.0, 10.0, 20.0])
            buffer_size = random.choice([50, 100, 150])
            nodes.append(NetworkNode(
                f"{node_type}-{i}", 
                (x, y, z),
                comm_range=random.uniform(80, 150),
                bandwidth=bandwidth,
                buffer_size=buffer_size
            ))
        return nodes
    
    def _initialize_links(self):
        links = []
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes[i+1:]):
                dist = self._distance(i, j+i+1)
                if dist < min(node1.comm_range, node2.comm_range):
                    latency = dist * 0.1 + random.uniform(0, 5)
                    bandwidth = min(node1.bandwidth, node2.bandwidth) * 0.8
                    reliability = max(0.7, 1.0 - (dist / min(node1.comm_range, node2.comm_range)))
                    links.append(NetworkLink(
                        node1.id, node2.id, latency, bandwidth, reliability=reliability
                    ))
        return links
    
    def _initialize_pheromones(self):
        for link in self.links:
            self.pheromone_matrix[link.source][link.target] = 0.1
            self.pheromone_matrix[link.target][link.source] = 0.1
    
    def update_network(self):
        # Node movement
        for node in self.nodes[1:]:
            if node.status == "active":
                dx, dy = random.uniform(-5, 5), random.uniform(-5, 5)
                dz = random.uniform(-2, 2) if "Drone" in node.id else 0
                node.position = (
                    node.position[0] + dx,
                    node.position[1] + dy,
                    max(0, node.position[2] + dz))
        
        # Update links based on new positions
        self.links = self._initialize_links()
        self._initialize_pheromones()
        
        # Random jamming effects
        if random.random() < 0.1:
            x, y = random.uniform(-150, 150), random.uniform(-150, 150)
            jam_power = random.uniform(0.5, 1.5)
            self.jamming_zones.append((x, y, random.uniform(30, 100), jam_power))
            
            for node in self.nodes:
                if node.status == "active":
                    dist = ((node.position[0]-x)**2 + (node.position[1]-y)**2)**0.5
                    if dist < self.jamming_zones[-1][2]:
                        node.status = "jammed"
                        for link in self.links:
                            if link.source == node.id or link.target == node.id:
                                link.reliability = max(0.3, link.reliability - 0.3 * jam_power)
        
        # Update node loads and delays
        for node in self.nodes:
            if node.status == "active":
                # Simulate traffic variation with some packets being processed
                processed_packets = min(node.buffer_occupancy, random.randint(0, 5))
                node.buffer_occupancy -= processed_packets
                
                # Add new packets based on current load
                new_packets = random.randint(0, int(node.current_load))
                node.buffer_occupancy = min(node.buffer_size, node.buffer_occupancy + new_packets)
                
                # Calculate queueing delay based on buffer occupancy
                node.queue_delay = (node.buffer_occupancy / node.buffer_size) * 50  # 0-50ms delay
        
        # Update link utilizations based on active flows
        for link in self.links:
            link.current_utilization = 0.0  # Reset before recalculation
            
        for flow in self.flows:
            if flow.active and flow.route:
                for i in range(len(flow.route)-1):
                    src, dst = flow.route[i], flow.route[i+1]
                    link = self._find_link(src, dst)
                    if link:
                        link.current_utilization += flow.required_bandwidth / link.max_bandwidth
        
        # Random recovery from jamming
        for node in self.nodes:
            if node.status == "jammed" and random.random() < 0.3:
                node.status = "active"
                for link in self.links:
                    if link.source == node.id or link.target == node.id:
                        link.reliability = min(0.95, link.reliability + 0.3)
    
    def _find_link(self, src, dst):
        for link in self.links:
            if (link.source == src and link.target == dst) or (link.source == dst and link.target == src):
                return link
        return None
    
    def _distance(self, i: int, j: int) -> float:
        a, b = self.nodes[i].position, self.nodes[j].position
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**0.5
    
    def add_flow(self, source: str, destination: str, traffic_class: str, bandwidth: float):
        """Add a new flow with admission control"""
        # Check if source and destination exist and are active
        src_node = next((n for n in self.nodes if n.id == source), None)
        dst_node = next((n for n in self.nodes if n.id == destination), None)
        
        if not src_node or not dst_node or src_node.status != "active" or dst_node.status != "active":
            return False
        
        # Check if requested bandwidth is available at source
        if src_node.current_load + bandwidth > src_node.bandwidth * MAX_CAPACITY_UTILIZATION:
            return False
        
        max_latency = {
            "command": 100,
            "video": 200,
            "sensor": 150,
            "data": 300
        }.get(traffic_class, 300)
        
        flow = NetworkFlow(
            source=source,
            destination=destination,
            traffic_class=traffic_class,
            required_bandwidth=bandwidth,
            max_latency=max_latency
        )
        
        # Find path using ACO
        path = self.find_optimal_path_aco(source, destination, traffic_class, bandwidth)
        if not path:
            return False
        
        flow.route = path
        flow.active = True
        
        # Reserve resources
        src_node.current_load += bandwidth
        for i in range(len(path)-1):
            link = self._find_link(path[i], path[i+1])
            if link:
                link.current_utilization += bandwidth / link.max_bandwidth
        
        self.flows.append(flow)
        return True
    
    def remove_flow(self, flow_index: int):
        """Remove a flow and release its resources"""
        if 0 <= flow_index < len(self.flows):
            flow = self.flows[flow_index]
            if flow.active:
                # Release node resources
                src_node = next(n for n in self.nodes if n.id == flow.source)
                src_node.current_load -= flow.required_bandwidth
                
                # Release link resources
                for i in range(len(flow.route)-1):
                    link = self._find_link(flow.route[i], flow.route[i+1])
                    if link:
                        link.current_utilization -= flow.required_bandwidth / link.max_bandwidth
            
            del self.flows[flow_index]
            return True
        return False
    
    def find_optimal_path_aco(self, start_id: str, end_id: str, traffic_class: str, bandwidth: float) -> List[str]:
        """Find optimal path using Ant Colony Optimization"""
        if start_id == end_id:
            return []
        
        priority = self.priority_classes.get(traffic_class, 3)
        min_reliability = MIN_RELIABILITY
        max_latency = MAX_LATENCY
        
        best_path = None
        best_cost = float('inf')
        
        for _ in range(ACO_ITERATIONS):
            paths = []
            path_costs = []
            
            # Generate paths for each ant
            for _ in range(NUM_ANTS):
                path = self._construct_ant_path(start_id, end_id, bandwidth, min_reliability, max_latency, priority)
                if path:
                    cost = self._calculate_path_cost(path, bandwidth, priority)
                    paths.append(path)
                    path_costs.append(cost)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
            
            # Update pheromones
            self._update_pheromones(paths, path_costs)
        
        return best_path
    
    def _construct_ant_path(self, current_id: str, end_id: str, bandwidth: float, 
                           min_reliability: float, max_latency: float, priority: int) -> List[str]:
        """Construct a path for a single ant"""
        path = [current_id]
        visited = set([current_id])
        total_latency = 0.0
        
        while current_id != end_id:
            current_node = next(n for n in self.nodes if n.id == current_id)
            neighbors = []
            probabilities = []
            
            # Get all possible next nodes
            for link in [l for l in self.links if l.source == current_id or l.target == current_id]:
                neighbor_id = link.target if link.source == current_id else link.source
                if neighbor_id in visited:
                    continue
                
                neighbor_node = next(n for n in self.nodes if n.id == neighbor_id)
                if neighbor_node.status != "active":
                    continue
                
                # Check constraints
                if (link.max_bandwidth * (1 - link.current_utilization) < bandwidth):
                    continue
                
                if link.reliability < min_reliability:
                    continue
                
                # Calculate heuristic value
                heuristic = self._calculate_heuristic(link, neighbor_node, end_id, priority)
                
                # Get pheromone value
                pheromone = self.pheromone_matrix[current_id].get(neighbor_id, 0.1)
                
                # Calculate selection probability
                probability = (pheromone ** ALPHA) * (heuristic ** BETA)
                
                neighbors.append(neighbor_id)
                probabilities.append(probability)
            
            if not neighbors:
                return []  # Dead end
            
            # Normalize probabilities
            prob_sum = sum(probabilities)
            if prob_sum == 0:
                probabilities = [1/len(neighbors)] * len(neighbors)
            else:
                probabilities = [p/prob_sum for p in probabilities]
            
            # Select next node
            next_id = np.random.choice(neighbors, p=probabilities)
            link = self._find_link(current_id, next_id)
            
            # Check if adding this link would exceed latency constraints
            if total_latency + link.latency > MAX_LATENCY:
                return []
            
            total_latency += link.latency
            path.append(next_id)
            visited.add(next_id)
            current_id = next_id
        
        return path
    
    def _calculate_heuristic(self, link: NetworkLink, neighbor_node: NetworkNode, 
                           end_id: str, priority: int) -> float:
        """Calculate heuristic value for ACO"""
        end_node = next(n for n in self.nodes if n.id == end_id)
        
        # Distance to destination (closer is better)
        dist_to_end = ((neighbor_node.position[0] - end_node.position[0])**2 +
                      (neighbor_node.position[1] - end_node.position[1])**2 +
                      (neighbor_node.position[2] - end_node.position[2])**2)**0.5
        
        # Current link metrics
        reliability_factor = link.reliability
        latency_factor = 1 / (link.latency + 0.1)
        utilization_factor = 1 - link.current_utilization
        queue_factor = 1 / (neighbor_node.queue_delay + 0.1)
        
        # Priority-based weighting
        priority_weight = 1 / (priority + 1)
        
        heuristic = (reliability_factor * 0.3 + 
                    latency_factor * 0.2 + 
                    utilization_factor * 0.2 + 
                    queue_factor * 0.2 + 
                    (1 / (dist_to_end + 1)) * 0.1) * priority_weight
        
        return max(0.001, heuristic)
    
    def _calculate_path_cost(self, path: List[str], bandwidth: float, priority: int) -> float:
        """Calculate total cost of a path"""
        if len(path) < 2:
            return float('inf')
        
        total_latency = 0.0
        min_reliability = 1.0
        max_utilization = 0.0
        hop_count = len(path) - 1
        
        for i in range(len(path)-1):
            src = path[i]
            dst = path[i+1]
            link = self._find_link(src, dst)
            
            if not link:
                return float('inf')
            
            src_node = next(n for n in self.nodes if n.id == src)
            dst_node = next(n for n in self.nodes if n.id == dst)
            
            total_latency += link.latency + src_node.queue_delay + dst_node.queue_delay
            min_reliability = min(min_reliability, link.reliability)
            max_utilization = max(max_utilization, link.current_utilization)
        
        # Cost components
        latency_cost = total_latency / MAX_LATENCY
        reliability_cost = (1 - min_reliability) * 2
        utilization_cost = max_utilization * 3
        hop_cost = hop_count * 0.1
        priority_cost = priority * 0.5
        
        total_cost = (latency_cost + reliability_cost + utilization_cost + 
                     hop_cost + priority_cost)
        
        return total_cost
    
    def _update_pheromones(self, paths: List[List[str]], path_costs: List[float]):
        """Update pheromone trails based on ant paths"""
        # Evaporate all pheromones
        for src in self.pheromone_matrix:
            for dst in self.pheromone_matrix[src]:
                self.pheromone_matrix[src][dst] *= (1 - EVAPORATION_RATE)
        
        # Deposit pheromones on used paths
        for path, cost in zip(paths, path_costs):
            if not path:
                continue
                
            deposit = PHEROMONE_DEPOSIT / (cost + 0.1)
            
            for i in range(len(path)-1):
                src = path[i]
                dst = path[i+1]
                
                if dst in self.pheromone_matrix.get(src, {}):
                    self.pheromone_matrix[src][dst] += deposit
                if src in self.pheromone_matrix.get(dst, {}):
                    self.pheromone_matrix[dst][src] += deposit

class JammingDetector:
    def detect(self, network) -> Dict[str, List[str]]:
        results = {
            "jammed_nodes": [],
            "congested_nodes": [],
            "congested_links": [],
            "critical_nodes": []
        }
        
        # Detect jammed nodes
        results["jammed_nodes"] = [n.id for n in network.nodes if n.status == "jammed"]
        
        # Detect congested nodes (high buffer occupancy)
        results["congested_nodes"] = [
            n.id for n in network.nodes 
            if n.status == "active" and n.buffer_occupancy / n.buffer_size > 0.8
        ]
        
        # Detect congested links
        results["congested_links"] = [
            f"{l.source}-{l.target}" 
            for l in network.links 
            if l.current_utilization > MAX_CAPACITY_UTILIZATION
        ]
        
        # Detect critical nodes (high degree and high load)
        degree = defaultdict(int)
        for link in network.links:
            degree[link.source] += 1
            degree[link.target] += 1
            
        results["critical_nodes"] = [
            n.id for n in network.nodes 
            if degree.get(n.id, 0) >= 3 and 
               n.current_load / n.bandwidth > 0.7
        ]
        
        return results

# ================== Streamlit UI ==================
if 'network' not in st.session_state:
    st.session_state.network = MilitaryNetwork(20)
    st.session_state.detector = JammingDetector()
    st.session_state.running = False
    st.session_state.show_jamming = True
    st.session_state.show_congestion = True
    st.session_state.show_flows = True
    st.session_state.next_flow_id = 1

st.set_page_config(layout="wide")
st.title("🛡️ Tactical Network Optimization with ACO (CNCMP-Compliant)")

# Control Panel
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶ Start Simulation"):
        st.session_state.running = True
with col2:
    if st.button("⏹ Stop Simulation"):
        st.session_state.running = False
with col3:
    st.session_state.traffic_class = st.selectbox(
        "Traffic Class",
        ["command", "video", "sensor", "data"]
    )

# Sidebar Controls
with st.sidebar:
    st.subheader("Network Controls")
    st.session_state.show_jamming = st.checkbox("Show Jamming Zones", True)
    st.session_state.show_congestion = st.checkbox("Show Congestion", True)
    st.session_state.show_flows = st.checkbox("Show Active Flows", True)
    
    st.subheader("Flow Management")
    source = st.selectbox("Source Node", [n.id for n in st.session_state.network.nodes])
    target = st.selectbox("Target Node", 
                         [n.id for n in st.session_state.network.nodes if n.id != source])
    bandwidth = st.slider("Bandwidth (Mbps)", 0.1, 10.0, 1.0)
    
    if st.button("Add Flow"):
        if st.session_state.network.add_flow(source, target, st.session_state.traffic_class, bandwidth):
            st.sidebar.success(f"Added {st.session_state.traffic_class} flow: {source} → {target}")
        else:
            st.sidebar.error("Failed to add flow (insufficient resources or no path)")
    
    if st.button("Detect Issues"):
        issues = st.session_state.detector.detect(st.session_state.network)
        if any(issues.values()):
            st.sidebar.warning("🚨 Network Issues Detected:")
            if issues["jammed_nodes"]:
                st.sidebar.write(f"Jammed Nodes: {', '.join(issues['jammed_nodes'])}")
            if issues["congested_nodes"]:
                st.sidebar.write(f"Congested Nodes: {', '.join(issues['congested_nodes'])}")
            if issues["congested_links"]:
                st.sidebar.write(f"Congested Links: {', '.join(issues['congested_links'])}")
            if issues["critical_nodes"]:
                st.sidebar.write(f"Critical Nodes: {', '.join(issues['critical_nodes'])}")
        else:
            st.sidebar.success("✅ No network issues detected")
    
    st.subheader("Active Flows")
    for i, flow in enumerate(st.session_state.network.flows):
        cols = st.columns([4, 1])
        cols[0].write(f"{flow.source} → {flow.destination} ({flow.traffic_class}, {flow.required_bandwidth:.1f}Mbps)")
        if cols[1].button("X", key=f"del_{i}"):
            st.session_state.network.remove_flow(i)
            st.rerun()

# Visualization Functions
def draw_network():
    fig, ax = plt.subplots(figsize=(14, 10))
    network = st.session_state.network
    
    # Draw jamming zones if enabled
    if st.session_state.show_jamming:
        for i, (x, y, r, power) in enumerate(network.jamming_zones):
            circle = plt.Circle((x, y), r, color='red', alpha=0.15*(i+1))
            ax.add_patch(circle)
            ax.text(x, y, f"JAM ZONE {i+1}\nPower: {power:.1f}", 
                   ha='center', va='center', color='darkred', weight='bold')
    
    # Draw all links first (background)
    for link in network.links:
        src_node = next(n for n in network.nodes if n.id == link.source)
        tgt_node = next(n for n in network.nodes if n.id == link.target)
        
        if src_node.status == "active" and tgt_node.status == "active":
            # Color based on utilization
            if st.session_state.show_congestion:
                if link.current_utilization > MAX_CAPACITY_UTILIZATION:
                    color = 'red'
                elif link.current_utilization > MAX_CAPACITY_UTILIZATION * 0.7:
                    color = 'orange'
                else:
                    color = 'green'
                alpha = 0.7
            else:
                color = 'gray'
                alpha = 0.3
            
            linewidth = 1 + 3 * link.current_utilization
            ax.plot(
                [src_node.position[0], tgt_node.position[0]],
                [src_node.position[1], tgt_node.position[1]], 
                color=color, alpha=alpha, linewidth=linewidth
            )
    
    # Highlight active flows if enabled
    if st.session_state.show_flows:
        for flow in network.flows:
            if flow.active and flow.route:
                path_x = []
                path_y = []
                for node_id in flow.route:
                    node = next(n for n in network.nodes if n.id == node_id)
                    path_x.append(node.position[0])
                    path_y.append(node.position[1])
                
                color = {
                    "command": "cyan",
                    "video": "magenta",
                    "sensor": "yellow",
                    "data": "lime"
                }.get(flow.traffic_class, "white")
                
                ax.plot(path_x, path_y, color=color, linewidth=3, alpha=0.7, linestyle='--')
    
    # Draw nodes on top
    for node in network.nodes:
        marker = '$V$' if "Vehicle" in node.id else '$D$' if "Drone" in node.id else 'o'
        
        # Node color based on status
        if node.status != "active":
            color = 'red'
        elif node.id == "Command":
            color = 'green'
        elif node.id in [f.source for f in network.flows] + [f.destination for f in network.flows]:
            color = 'purple'
        else:
            color = 'blue'
        
        # Node size based on buffer occupancy
        size = 100 + (node.buffer_occupancy / node.buffer_size) * 200
        
        ax.scatter(node.position[0], node.position[1], 
                  marker=marker, c=color, s=size, edgecolors='black', linewidths=1, zorder=20)
        
        # Node info text
        info_text = f"{node.id}\nLoad: {node.current_load:.1f}/{node.bandwidth:.1f}Mbps\nBuf: {node.buffer_occupancy}/{node.buffer_size}"
        ax.text(node.position[0], node.position[1]+15, info_text, 
               ha='center', fontsize=7, weight='bold', zorder=30)
    
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_title("Tactical Network Status", pad=20)
    ax.grid(True, alpha=0.2)
    return fig

# Main display
st.pyplot(draw_network())

# Network statistics
st.subheader("Network Statistics")
col1, col2, col3, col4 = st.columns(4)
active_nodes = sum(1 for n in st.session_state.network.nodes if n.status == "active")
jammed_nodes = sum(1 for n in st.session_state.network.nodes if n.status == "jammed")
active_flows = sum(1 for f in st.session_state.network.flows if f.active)
congested_links = sum(1 for l in st.session_state.network.links if l.current_utilization > MAX_CAPACITY_UTILIZATION)

col1.metric("Active Nodes", active_nodes)
col2.metric("Jammed Nodes", jammed_nodes)
col3.metric("Active Flows", active_flows)
col4.metric("Congested Links", congested_links)

# Simulation Update
if st.session_state.running:
    st.session_state.network.update_network()
    time.sleep(0.5)
    st.rerun()