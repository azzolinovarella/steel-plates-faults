# Bibliotecas de terceiros
## Utilitárias
import numpy as np
## Para gráficos e visualização
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importações de constantes personalizadas
from useful.constants import PALETTE, BLUE, RED, N_ITER, PLOTLY_TEMPLATE, FONT_SIZE

    
def generate_colors(num_colors):
    colors = px.colors.sample_colorscale(PALETTE, [n/(num_colors - 1) for n in range(num_colors)])
    
    return colors


def plot_accuracies(res, export=False, filename='acc', path='/tmp'):
    accuracies = [r*100 for r in res['accuracies']]  # Acho que fica melhor em percentual
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=accuracies, nbinsx=5, marker_color=BLUE))
    fig.add_vline(x=np.mean(accuracies), line_dash='dash', annotation_text=f' Acurácia Média: {np.mean(accuracies):.2f}%')
    fig.update_layout(title=f'Distribuição das Acurácias após {N_ITER} iterações', yaxis_title='Frequência', xaxis_title='Acurácia (%)',
                      height=600, autosize=True)

    if export:
        export_fig(fig, filename, path)
    
    fig.show()


def plot_label_metrics(res, metric, export=False, filename='label_metric', path='/tmp'):
    label_score = {k: [v*100 for v in res[metric][k]] for k in res[metric].keys()}  # Percentual
    
    labels = list(label_score.keys())
    marker_colors = generate_colors(len(labels))

    fig = go.Figure()
    for i, l in enumerate(labels):
        color = marker_colors[i]
        f1_score = label_score[l]

        fig.add_trace(go.Box(y=f1_score, name=l, marker_color=color, legendgroup=i))#, boxpoints='all'))

    fig.update_layout(title=f'Boxplots de {metric.title()} por label após {N_ITER} iterações', yaxis_title=f"{metric.title()} (%)",
                      xaxis_title="Classes", yaxis_range=(0, 101), height=600, autosize=True)  # Para ter comparabilidade entre os modelos

    if export:
        export_fig(fig, filename, path)
    
    fig.show()


def plot_confusion_matrix(res, export=False, filename='cm', path='/tmp'):
    cms = res['cms']
    axis_labels = list(res['recalls'].keys())  # Igual para todos
    
    cm = np.sum(cms, axis=0)
    cm_mean = np.mean(cms, axis=0)
    cm_recall = cm/cm.sum(axis=1, keepdims=True)  # Normalização pela linha
    cm_precision = cm/cm.sum(axis=0, keepdims=True)  # Normalização pela coluna

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.1,
                        subplot_titles=['Matriz de Confusão agregada', 'Matriz de Confusão média', 
                                        'Matriz de Confusão normalizada pelas linhas', 
                                        'Matriz de Confusão normalizada pelas colunas'])
    
    fig.add_trace(go.Heatmap(x=axis_labels, y=axis_labels, z=cm, text=cm, texttemplate='%{text}', 
                             showscale=False, colorscale=PALETTE), 
                  row=1, col=1)
    fig.add_trace(go.Heatmap(x=axis_labels, y=axis_labels, z=cm_mean, text=cm_mean, texttemplate='%{text:.2f}', 
                             showscale=False, colorscale=PALETTE), 
                  row=1, col=2)


    fig.add_trace(go.Heatmap(x=axis_labels, y=axis_labels, z=cm_recall, text=cm_recall, texttemplate='%{text:.2f}', 
                             showscale=False, colorscale=PALETTE), 
                  row=2, col=1)
    fig.add_trace(go.Heatmap(x=axis_labels, y=axis_labels, z=cm_precision, text=cm_precision, texttemplate='%{text:.2f}', 
                             showscale=False, colorscale=PALETTE), 
                  row=2, col=2)

    fig.update_layout(title=f'Matrizes de Confusão geradas após {N_ITER} iterações', yaxis1_title='Real', yaxis3_title='Real', 
                      xaxis3_title='Predito', xaxis4_title='Predito', yaxis1_autorange='reversed', yaxis2_autorange='reversed', 
                      yaxis3_autorange='reversed', height=1000, autosize=True)

    if export:
        export_fig(fig, filename, path)

    fig.show()


# def plot_comparative_metric(metrics_dict):
#     fig = make_subplots(rows=len(metrics_dict), cols=1, shared_xaxes=True, shared_yaxes=True, 
#                         subplot_titles=list(metrics_dict.keys()))

#     for i, (approach, accuracies) in enumerate(metrics_dict.items()):
#         fig.add_trace(go.Histogram(x=accuracies, name=approach, marker_color=BLUE),
#                       row=i + 1, col=1)

#         mean_acc = np.mean(accuracies)
#         fig.add_vline(x=mean_acc, annotation_text=f'  Valor médio: {mean_acc:.2f}%', line_dash='dash', line_color='black',
#                      row=i + 1, col=1)

#     fig.update_layout(title=f'Distribuição das Acurácias por abordagem', yaxis1_title='Frequência', yaxis2_title='Frequência',
#                       yaxis3_title='Frequência', yaxis4_title='Frequência', xaxis4_title='Acurácia (%)', height=800,  
#                       autosize=True, barmode='overlay', yaxis1_range=(0, 12), yaxis2_range=(0, 12), yaxis3_range=(0, 12), 
#                       yaxis4_range=(0, 12), showlegend=False)

#     fig.show()


def plot_comparative_metric(metrics_dict, metric_name, key_to_compare, export=False, filename='metric_comp', path='/tmp'):
    fig = go.Figure()

    approaches = {k: v for k, v in metrics_dict.items() if k != key_to_compare}
    for approach in approaches:
        metric = metrics_dict[approach]
        
        fig.add_trace(go.Violin(y=metrics_dict[key_to_compare], name=approach, marker_color=RED, side='negative', box_visible=True,
                                meanline_visible=True))
        fig.add_trace(go.Violin(y=metric, name=approach, marker_color=BLUE, side='positive', box_visible=True, meanline_visible=True))

    fig.update_layout(title=f'Comparação de {metric_name}s sem bagging e com bagging<br><sup>Cor vermelha representando'
                      f'a estratégia sem bagging</sup>', 
                      yaxis_title=f'{metric_name.title()} (%)', 
                      height=600,  autosize=True, showlegend=False)
    
    if export:
        export_fig(fig, filename, path)

    fig.show()


def export_fig(fig, filename, path):
    pio.templates['exporter'] = pio.templates[PLOTLY_TEMPLATE]
    pio.templates['exporter']['layout']['font']['size'] = FONT_SIZE  # Usado para exportar para png
    
    fig_png = go.Figure(fig)  # Deep copy para não alterar o objeto original
    fig_html = go.Figure(fig)

    fig_png = fig_png.update_layout(template='exporter', width=1100)  # Para cobrir toda lateral da pag
    fig_html = fig_html.update_layout(width=None, height=None, autosize=True)  # Para mudar conforme a página html

    fig_png.write_image(f'{path}/{filename}.png', scale=3)
    fig_html.write_html(f'{path}/{filename}.html')