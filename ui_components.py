import plotly.graph_objects as go
BLUE_PALETTE = ["#2563EB", "#60A5FA", "#93C5FD", "#1E40AF"]
def create_score_gauge(score):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, number={"suffix": "%"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563EB"}}))
    fig.update_layout(height=260, margin=dict(l=20,r=20,t=30,b=10))
    return fig
def create_radar_competency_chart(subscores):
    labels = list(subscores)
    values = list(subscores.values())
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill="toself"))
    fig.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 100]}}, height=350)
    return fig
def create_skills_category_bar_chart(categories):
    fig = go.Figure(go.Bar(x=list(categories), y=[len(v) for v in categories.values()], marker_color="#2563EB"))
    fig.update_layout(height=300, yaxis_title="Skills detected")
    return fig
def create_confusion_matrix_heatmap(matrix):
    fig = go.Figure(go.Heatmap(z=matrix, x=["Predicted no", "Predicted yes"], y=["Actual no", "Actual yes"], colorscale="Blues", showscale=False, text=matrix, texttemplate="%{text}"))
    fig.update_layout(height=300)
    return fig
def create_roc_curve_chart(*args, **kwargs):
    return go.Figure()
def create_ann_training_history_chart(history):
    fig = go.Figure(go.Scatter(y=history.get("loss", []), mode="lines", name="Training loss"))
    fig.update_layout(xaxis_title="Iteration", yaxis_title="Loss", height=300)
    return fig
