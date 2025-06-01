import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt  # 用于颜色映射

# ======================
# 虚拟数据生成模块
# ======================
def generate_structured_data(nodes):
    """生成具有群落结构的虚拟数据"""
    n = len(nodes)
    matrix = np.zeros((n, n))
    
    # 定义三个群落分组
    groups = [nodes[:2], nodes[2:4], nodes[4:]]
    
    # 组内强连接
    for group in groups:
        for i in group:
            for j in group:
                if i != j:
                    idx_i = nodes.index(i)
                    idx_j = nodes.index(j)
                    matrix[idx_i][idx_j] = np.random.randint(5, 10)
    
    # 组间弱连接
    matrix[0,4] = matrix[4,0] = 3  # 跨群组连接
    matrix[1,3] = matrix[3,1] = 2
    matrix[2,5] = matrix[5,2] = 4
    
    np.fill_diagonal(matrix, 0)  # 清除自连接
    return matrix

# 节点配置
nodes = ['New York', 'London', 'Tokyo', 'Beijing', 'Paris', 'Sydney']

# ======================
# 数据格式转换
# ======================
matrix = generate_structured_data(nodes)
sources, targets, values = [], [], []
for i in range(len(nodes)):
    for j in range(len(nodes)):
        if i < j and matrix[i][j] > 0:  # 避免重复记录
            sources.append(i)
            targets.append(j)
            values.append(matrix[i][j])

# ======================
# 可视化引擎
# ======================
# 创建颜色映射
color_palette = plt.cm.tab20.colors  # 使用预定义颜色
node_colors = [color_palette[i % 20] for i in range(len(nodes))]

fig = go.Figure(go.Chord(
    nodes=[go.Node(
        label=node, 
        color=f'rgb{tuple(int(255*c) for c in color)}',  # 转换matplotlib颜色格式
        hovertemplate="%{label}<extra></extra>")  # 自定义悬停信息
        for node, color in zip(nodes, node_colors)],
    
    links=dict(
        source=sources,
        target=targets,
        value=values,
        color=[node_colors[s] for s in sources],  # 按源节点着色
        hovertemplate="%{source.label} → %{target.label}<br>强度: %{value}<extra></extra>"
    ),
    
    # 布局参数
    opacity=0.8,
    gap=0.05,  # 节点间距
    show_ticks=False
))

# ======================
# 图表布局配置
# ======================
fig.update_layout(
    title_text="全球城市关联网络",
    title_x=0.5,
    title_font_size=24,
    margin=dict(t=100, b=20, l=20, r=20),
    paper_bgcolor='rgba(245,245,245,0.9)',
    hoverlabel=dict(
        bgcolor="white",
        font_size=16
    ),
    annotations=[
        dict(
            text="数据说明：线宽表示连接强度，颜色表示来源城市",
            x=0.5,
            y=-0.1,
            showarrow=False,
            font=dict(color="gray")
        )
    ]
)

# ======================
# 交互功能增强
# ======================
fig.update_traces(
    node_hoverinfo='label',
    link_hoverinfo='none'  # 禁用默认悬停显示
)

# 显示图表
fig.show()

# 保存选项（取消注释使用）
# fig.write_html("interactive_chord.html")
# fig.write_image("chord.png", scale=2)