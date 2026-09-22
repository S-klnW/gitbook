import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import PathPatch

TREE = {'flare': {
    'analytics': {
        'cluster': dict.fromkeys(['AgglomerativeCluster', 'CommunityStructure',
                                  'HierarchicalCluster', 'MergeEdge']),
        'graph': dict.fromkeys(['BetweennessCentrality', 'LinkDistance',
                                'MaxFlowMinCut', 'ShortestPaths', 'SpanningTree']),
        'optimization': {'AspectRatioBanker': None},
    },
    'animate': dict.fromkeys(['Easing', 'FunctionSequence', 'IInterpolator',
                              'ISchedulable', 'Parallel', 'Pause', 'Scheduler',
                              'Sequence', 'Transition', 'Transitioner', 'Tween']),
    'data': {
        'converters': dict.fromkeys(['Converters', 'DelimitedTextConverter',
                                     'GraphMLConverter', 'IDataConverter',
                                     'JSONConverter']),
        'DataField': None, 'DataSchema': None, 'DataSet': None,
        'DataSource': None, 'DataTable': None, 'DataUtil': None,
    },
    'display': dict.fromkeys(['DirtySprite', 'LineSprite', 'RectSprite', 'TextSprite']),
    'flex': {'FlareVis': None},
    'physics': dict.fromkeys(['DragForce', 'GravityForce', 'IForce', 'NBodyForce',
                              'Particle', 'Simulation', 'Spring', 'SpringForce']),
    'query': {
        'methods': dict.fromkeys(['And', 'Arithmetic', 'Average', 'BinaryExpression',
                                  'Comparison', 'CompositeExpression', 'Count',
                                  'DateUtil', 'Distinct', 'Expression',
                                  'ExpressionIterator', 'Fn', 'If', 'IsA', 'Literal',
                                  'Match', 'Maximum', 'Minimum', 'Not', 'Or', 'Query',
                                  'Range', 'StringUtil', 'Sum', 'Variable', 'Variance']),
    },
    'scale': dict.fromkeys(['IScaleMap', 'LinearScale', 'LogScale', 'OrdinalScale',
                            'QuantileScale', 'QuantitativeScale', 'RootScale',
                            'ScaleType', 'TimeScale']),
    'util': {
        'heap': dict.fromkeys(['FibonacciHeap', 'HeapNode']),
        'Arrays': None, 'Colors': None, 'Dates': None, 'Displays': None,
        'Filter': None, 'Geometry': None, 'IEvaluable': None, 'Maths': None,
        'Orientation': None, 'Property': None, 'Shapes': None, 'Sort': None,
        'Stats': None, 'Strings': None,
    },
    'vis': {
        'axis': dict.fromkeys(['Axes', 'Axis', 'AxisGridLine', 'AxisLabel', 'CartesianAxes']),
        'controls': dict.fromkeys(['AnchorControl', 'ClickControl', 'Control',
                                   'ControlList', 'DragControl', 'ExpandControl',
                                   'HoverControl', 'PanZoomControl',
                                   'SelectionControl', 'TooltipControl']),
        'data': dict.fromkeys(['DataList', 'DataSprite', 'EdgeSprite', 'NodeSprite',
                               'ScaleBinding', 'Tree', 'TreeBuilder']),
        'events': dict.fromkeys(['DataEvent', 'SelectionEvent', 'TooltipEvent',
                                 'VisualizationEvent']),
        'legend': dict.fromkeys(['Legend', 'LegendItem', 'LegendRange']),
        'operator': {
            'distortion': dict.fromkeys(['BifocalDistortion', 'FisheyeDistortion']),
            'encoder': dict.fromkeys(['ColorEncoder', 'Encoder', 'PropertyEncoder']),
            'label': dict.fromkeys(['Labeler', 'StackedAreaLabel']),
            'Operator': None, 'OperatorList': None,
            'OperatorSequence': None, 'SortOperator': None,
        },
        'Visualization': None,
    },
}}


def count_leaves(node):
    return 1 if not node else sum(count_leaves(v) for v in node.values())

N_LEAF = count_leaves(TREE['flare'])
nodes, edges, counter = [], [], [0]

def walk(name, node, depth):
    is_leaf = not node                       
    if node:
        res = [walk(c, node[c], depth + 1) for c in node]
        theta = float(np.mean([w[0] for w in res]))
        for w in res:
            edges.append((theta, w[0], depth, depth + 1))
    else:
        theta = -np.pi / 2 + (counter[0] + 0.5) / N_LEAF * 2 * np.pi
        counter[0] += 1
    nodes.append((name, theta, depth, is_leaf))
    return (theta, depth, name, depth)

walk('flare', TREE['flare'], 0)
DEPTH = max(d for _, _, d, _ in nodes)
N_LEAF_ACTUAL = sum(1 for _, _, _, lf in nodes if lf)
print(f'节点 {len(nodes)} | 叶子 {N_LEAF_ACTUAL} | 最大深度 {DEPTH}')

FIG, R = 16, 1.0
fig, ax = plt.subplots(figsize=(FIG, FIG))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
P = lambda th, r: (r * np.cos(th), r * np.sin(th))

WHITE = [pe.withStroke(linewidth=1.6, foreground='white')]

# (a) 连线
for t1, t2, d1, d2 in edges:
    x1, y1 = P(t1, d1 * R); x2, y2 = P(t2, d2 * R)
    rm = (d1 + d2) / 2 * R
    ax.add_patch(PathPatch(
        Path([(x1, y1), P(t1, rm), P(t2, rm), (x2, y2)],
             [Path.MOVETO] + [Path.CURVE4] * 3),
        fc='none', ec='#B9C4CC', lw=0.65, zorder=1))

# (b) 节点
for name, th, d, lf in nodes:
    if d == 0:
        ms, col = 6.5, '#2E6F9E'
    elif lf:
        ms, col = 2.6, '#8FC1E8'
    else:
        ms, col = 3.6, '#4A90C2'
    ax.plot(*P(th, d * R), marker='o', ms=ms, color=col, mec='none', zorder=3)

# (c) 全部节点标注
def put_label(name, th, d, offset, fs, color, z):
    if d == 0:                                   # 根节点：居中不旋转
        ax.text(0, -0.16, name, ha='center', va='center', fontsize=fs,
                color=color, path_effects=WHITE, zorder=z)
        return
    ang = np.degrees(th) % 360
    rot = ang + 180 if 90 < ang < 270 else ang
    ha  = 'right'   if 90 < ang < 270 else 'left'
    ax.text(*P(th, (d + offset) * R), name,
            rotation=rot, ha=ha, va='center', rotation_mode='anchor',
            fontsize=fs, color=color, path_effects=WHITE, zorder=z)

STYLE = {0: (0.00, 15, '#1F4E79'),   # 根
         1: (0.14, 12, '#2F4F6F'),   # 一级分支
         2: (0.075, 7.5, '#4A6A85'),
         3: (0.070, 6.2, '#4A6A85')}

for name, th, d, lf in nodes:
    if lf:                                       # 叶子：统一最外圈字号
        put_label(name, th, d, 0.10, 6.0, '#333333', 5)
    else:
        off, fs, col = STYLE.get(d, (0.07, 6.2, '#4A6A85'))
        put_label(name, th, d, off, fs, col, 4)

lim = (DEPTH + 1.50) * R
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect('equal'); ax.axis('off')
plt.title('Radial Tidy Tree - D3 flare (matplotlib, all nodes labeled)',
          pad=16, fontsize=15)
plt.tight_layout()
plt.savefig('radial_tidy_tree_flare_labeled.svg', format='svg',
            bbox_inches='tight', facecolor='white')
plt.savefig('radial_tidy_tree_flare_labeled.png', dpi=170,
            bbox_inches='tight', facecolor='white')
print('OK -> radial_tidy_tree_flare_labeled.svg / .png')
