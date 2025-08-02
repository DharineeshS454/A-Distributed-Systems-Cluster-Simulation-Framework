import streamlit as st
import requests
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import threading
import os

# Set page configuration
st.set_page_config(
    page_title="Kubernetes Cluster Simulation",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066CC;
        margin-bottom: 1rem;
    }
    .node-healthy {
        color: green;
        font-weight: bold;
    }
    .node-unhealthy {
        color: red;
        font-weight: bold;
    }
    .stat-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0066CC;
    }
    .stat-label {
        font-size: 1rem;
        color: #555;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_url' not in st.session_state:
    st.session_state.api_url = os.environ.get("API_SERVER_URL", "http://localhost:5001")
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5
if 'refresh_thread' not in st.session_state:
    st.session_state.refresh_thread = None
if 'refresh_active' not in st.session_state:
    st.session_state.refresh_active = False

# Functions to interact with the API
def get_nodes():
    try:
        response = requests.get(f"{st.session_state.api_url}/nodes")
        if response.status_code == 200:
            return response.json().get('nodes', {})
        else:
            st.error(f"Failed to fetch nodes: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error connecting to API server: {str(e)}")
        return {}

def get_pods():
    try:
        response = requests.get(f"{st.session_state.api_url}/pods")
        if response.status_code == 200:
            return response.json().get('pods', {})
        else:
            st.error(f"Failed to fetch pods: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error connecting to API server: {str(e)}")
        return {}

def get_stats():
    try:
        response = requests.get(f"{st.session_state.api_url}/stats")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch stats: {response.text}")
            return {'nodes': {}, 'pods': {}}
    except Exception as e:
        st.error(f"Error connecting to API server: {str(e)}")
        return {'nodes': {}, 'pods': {}}

def add_node(cpu_cores):
    try:
        response = requests.post(
            f"{st.session_state.api_url}/nodes",
            json={'cpu_cores': cpu_cores}
        )
        if response.status_code == 201:
            data = response.json()
            st.success(f"Node added successfully! Node ID: {data['node_id']}")
            return True
        else:
            st.error(f"Failed to add node: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error connecting to API server: {str(e)}")
        return False

def create_pod(cpu_requirement, algorithm):
    try:
        response = requests.post(
            f"{st.session_state.api_url}/pods",
            json={
                'cpu_requirement': cpu_requirement,
                'algorithm': algorithm
            }
        )
        if response.status_code == 201:
            data = response.json()
            st.success(f"Pod created successfully! Pod ID: {data['pod_id']}")
            return True
        else:
            st.error(f"Failed to create pod: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error connecting to API server: {str(e)}")
        return False

def auto_refresh():
    while st.session_state.refresh_active:
        # Update the dashboard data
        try:
            time.sleep(st.session_state.refresh_interval)
            st.session_state.last_update = datetime.now()
            st.experimental_rerun()
        except Exception as e:
            # Handle any errors that might occur during refresh
            pass

def toggle_auto_refresh():
    st.session_state.auto_refresh = not st.session_state.auto_refresh
    
    if st.session_state.auto_refresh:
        st.session_state.refresh_active = True
        refresh_thread = threading.Thread(target=auto_refresh)
        refresh_thread.daemon = True
        refresh_thread.start()
        st.session_state.refresh_thread = refresh_thread
    else:
        st.session_state.refresh_active = False
        # Let the thread terminate on its own by checking the refresh_active flag

# Sidebar for settings and actions
with st.sidebar:
    st.image("https://kubernetes.io/images/kubernetes-horizontal-color.png", width=250)
    st.markdown("## Settings")
    
    # API Server URL
    api_url = st.text_input("API Server URL", value=st.session_state.api_url)
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
    
    # Auto-refresh toggle
    st.checkbox("Auto-refresh dashboard", value=st.session_state.auto_refresh, 
                on_change=toggle_auto_refresh)
    
    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.slider(
            "Refresh interval (seconds)", 
            min_value=1, max_value=30, value=st.session_state.refresh_interval
        )
    
    st.markdown("---")
    
    # Add Node Form
    st.markdown("## Add Node")
    cpu_cores = st.number_input("CPU Cores", min_value=1, max_value=32, value=4, step=1)
    if st.button("Add Node", key="add_node_button"):
        add_node(int(cpu_cores))
    
    st.markdown("---")
    
    # Create Pod Form
    st.markdown("## Create Pod")
    cpu_requirement = st.number_input("CPU Requirement", min_value=1, max_value=16, value=1, step=1)
    algorithm = st.selectbox("Scheduling Algorithm", 
                           options=["first-fit", "best-fit", "worst-fit"],
                           index=0)
    if st.button("Create Pod", key="create_pod_button"):
        create_pod(int(cpu_requirement), algorithm)
    
    st.markdown("---")
    st.markdown(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    if not st.session_state.auto_refresh:
        if st.button("Refresh Dashboard"):
            st.session_state.last_update = datetime.now()
            st.experimental_rerun()

# Main content
st.markdown("<h1 class='main-header'>Kubernetes Cluster Simulation Dashboard</h1>", unsafe_allow_html=True)

# Get current data
nodes = get_nodes()
pods = get_pods()
stats = get_stats()

# Overview Stats in cards
st.markdown("## 📊 Cluster Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-value'>{stats.get('nodes', {}).get('total_nodes', 0)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-label'>Total Nodes</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-value'>{stats.get('pods', {}).get('total_pods', 0)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-label'>Total Pods</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    health_percentage = 0
    if stats.get('nodes', {}).get('total_nodes', 0) > 0:
        health_percentage = (stats.get('nodes', {}).get('healthy_nodes', 0) / stats.get('nodes', {}).get('total_nodes', 1)) * 100
    
    st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-value'>{health_percentage:.1f}%</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-label'>Cluster Health</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    utilization = 0
    if stats.get('nodes', {}).get('total_cores', 0) > 0:
        utilization = (stats.get('nodes', {}).get('used_cores', 0) / stats.get('nodes', {}).get('total_cores', 1)) * 100
    
    st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-value'>{utilization:.1f}%</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-label'>CPU Utilization</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Health and Resource Visualization
st.markdown("## 🔍 Health & Resources")
col1, col2 = st.columns(2)

with col1:
    # Node Health Pie Chart
    healthy_nodes = stats.get('nodes', {}).get('healthy_nodes', 0)
    unhealthy_nodes = stats.get('nodes', {}).get('unhealthy_nodes', 0)
    
    if healthy_nodes > 0 or unhealthy_nodes > 0:
        fig = go.Figure(data=[go.Pie(
            labels=['Healthy', 'Unhealthy'],
            values=[healthy_nodes, unhealthy_nodes],
            hole=.3,
            marker_colors=['#4CAF50', '#F44336']
        )])
        fig.update_layout(
            title_text="Node Health Distribution",
            width=800,
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("No nodes available for health visualization")

with col2:
    # CPU Resource Usage Bar Chart
    total_cores = stats.get('nodes', {}).get('total_cores', 0)
    used_cores = stats.get('nodes', {}).get('used_cores', 0)
    available_cores = stats.get('nodes', {}).get('available_cores', 0)
    
    if total_cores > 0:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['CPU Cores'],
            y=[used_cores],
            name='Used',
            marker_color='#FF9800'
        ))
        fig.add_trace(go.Bar(
            x=['CPU Cores'],
            y=[available_cores],
            name='Available',
            marker_color='#2196F3'
        ))
        fig.update_layout(
            barmode='stack',
            title_text="CPU Resource Allocation",
            width=800,
            height=500,
            xaxis=dict(title=''),
            yaxis=dict(title='Cores'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("No CPU resource data available")

# Nodes and Pods Tabs
st.markdown("## 📋 Cluster Resources")
tab1, tab2 = st.tabs(["Nodes", "Pods"])

with tab1:
    if nodes:
        # Create a dataframe for nodes
        node_data = []
        for node_id, node_info in nodes.items():
            node_data.append({
                'Node ID': node_id[:8] + '...' if len(node_id) > 10 else node_id,
                'Full ID': node_id,
                'CPU Cores': node_info['cpu_cores'],
                'Available Cores': node_info['available_cores'],
                'Status': node_info['status'],
                'Pod Count': len(node_info['pods'])
            })
        
        node_df = pd.DataFrame(node_data)
        
        # Apply conditional formatting
        def highlight_status(s):
            return ['background-color: #C8E6C9' if s == 'healthy' else 'background-color: #FFCDD2' for s in s]
        
        # Apply the formatting and display
        if not node_df.empty:
            st.dataframe(
                node_df.style.apply(lambda x: highlight_status(x) if x.name == 'Status' else [''] * len(x), axis=0),
                use_container_width=True
            )
            
            # Node details expander
            with st.expander("Node Details"):
                for node_id, node_info in nodes.items():
                    status_class = "node-healthy" if node_info['status'] == 'healthy' else "node-unhealthy"
                    st.markdown(f"#### Node: {node_id}")
                    st.markdown(f"- Status: <span class='{status_class}'>{node_info['status']}</span>", unsafe_allow_html=True)
                    st.markdown(f"- CPU Cores: {node_info['cpu_cores']}")
                    st.markdown(f"- Available Cores: {node_info['available_cores']}")
                    st.markdown(f"- Used Cores: {node_info['cpu_cores'] - node_info['available_cores']}")
                    st.markdown(f"- Pods: {', '.join(node_info['pods']) if node_info['pods'] else 'None'}")
                    st.markdown("---")
        else:
            st.info("No nodes found in the cluster")
    else:
        st.info("No nodes found in the cluster")

with tab2:
    if pods:
        # Create a dataframe for pods
        pod_data = []
        for pod_id, pod_info in pods.items():
            pod_data.append({
                'Pod ID': pod_id[:8] + '...' if len(pod_id) > 10 else pod_id,
                'Full ID': pod_id,
                'CPU Requirement': pod_info['cpu_requirement'],
                'Node ID': pod_info['node_id'][:8] + '...' if len(pod_info['node_id']) > 10 else pod_info['node_id'],
                'Node Status': nodes.get(pod_info['node_id'], {}).get('status', 'unknown')
            })
        
        pod_df = pd.DataFrame(pod_data)
        
        # Apply conditional formatting for node status
        def highlight_node_status(s):
            return ['background-color: #C8E6C9' if s == 'healthy' else 'background-color: #FFCDD2' for s in s]
        
        # Apply the formatting and display
        if not pod_df.empty:
            st.dataframe(
                pod_df.style.apply(lambda x: highlight_node_status(x) if x.name == 'Node Status' else [''] * len(x), axis=0),
                use_container_width=True
            )
            
            # Pod allocation visualization
            if nodes:
                pod_per_node = {}
                for node_id, node_info in nodes.items():
                    pod_per_node[node_id[:8] + '...'] = len(node_info['pods'])
                
                fig = px.bar(
                    x=list(pod_per_node.keys()),
                    y=list(pod_per_node.values()),
                    labels={'x': 'Node ID', 'y': 'Number of Pods'},
                    title="Pods per Node",
                    width=800,
                    height=500
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        else:
            st.info("No pods found in the cluster")
    else:
        st.info("No pods found in the cluster")

# Resource allocation visualization
st.markdown("## 📈 Resource Allocation")

if nodes:
    # Create data for node resource visualization
    node_resources = []
    for node_id, node_info in nodes.items():
        node_resources.append({
            'Node ID': node_id[:8] + '...',
            'Total Cores': node_info['cpu_cores'],
            'Used Cores': node_info['cpu_cores'] - node_info['available_cores'],
            'Available Cores': node_info['available_cores'],
            'Utilization (%)': ((node_info['cpu_cores'] - node_info['available_cores']) / node_info['cpu_cores']) * 100 if node_info['cpu_cores'] > 0 else 0
        })
    
    node_resource_df = pd.DataFrame(node_resources)
    
    if not node_resource_df.empty:
        # Bar chart for core allocation per node
        fig = go.Figure()
        
        for i, node in enumerate(node_resource_df['Node ID']):
            fig.add_trace(go.Bar(
                x=[node],
                y=[node_resource_df.loc[i, 'Used Cores']],
                name='Used',
                marker_color='#FF9800',
                showlegend=i==0
            ))
            fig.add_trace(go.Bar(
                x=[node],
                y=[node_resource_df.loc[i, 'Available Cores']],
                name='Available',
                marker_color='#2196F3',
                showlegend=i==0
            ))
        
        fig.update_layout(
            barmode='stack',
            title='CPU Resource Allocation per Node',
            width=800,
            height=500,
            xaxis=dict(
                title='Node ID',
                tickangle=-45
            ),
            yaxis=dict(title='CPU Cores'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Heat map for node utilization
        fig = go.Figure(data=go.Heatmap(
            z=[node_resource_df['Utilization (%)']],
            x=node_resource_df['Node ID'],
            y=['Utilization'],
            colorscale='RdYlGn_r',
            showscale=True
        ))
        
        fig.update_layout(
            title='CPU Utilization Heat Map',
            width=800,
            height=300,
            xaxis=dict(
                title='Node ID',
                tickangle=-45
            ),
            yaxis=dict(title='')
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("No node resource data available")
else:
    st.info("No nodes found in the cluster for resource allocation visualization")

# Footer
st.markdown("---")
st.markdown("<center>Kubernetes Cluster Simulation Dashboard | Created with Streamlit</center>", unsafe_allow_html=True)

# Handle auto-refresh thread termination when the app is closed
# This is handled by the daemon=True flag in the thread creation
