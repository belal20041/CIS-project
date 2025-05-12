import plotly.express as px
class EDAVisualizer:
    def plot_line(self, df, x, y, title):
        fig = px.line(df, x=x, y=y, title=title)
        return fig
