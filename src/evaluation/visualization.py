import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd


class FantasyVisualizer:
    def __init__(self):
        self.colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

    def plot_metrics_comparison(self, metrics_list, model_names):
        metrics_to_plot = ['validity_rate', 'avg_quality', 'combined_score']

        fig = go.Figure()

        for i, metrics in enumerate(metrics_list):
            fig.add_trace(go.Bar(
                x=metrics_to_plot,
                y=[metrics[m] for m in metrics_to_plot],
                name=model_names[i],
                marker_color=self.colors[i % len(self.colors)]
            ))

        fig.update_layout(
            title='Model Metrics Comparison',
            xaxis_title='Metrics',
            yaxis_title='Score',
            barmode='group'
        )

        return fig

    def plot_budget_utilization(self, metrics_list, model_names):
        fig = go.Figure()

        for i, metrics in enumerate(metrics_list):
            fig.add_trace(go.Bar(
                x=[model_names[i]],
                y=[metrics['avg_budget_utilization']],
                name=model_names[i],
                marker_color=self.colors[i % len(self.colors)],
                error_y=dict(
                    type='data',
                    array=[metrics['budget_utilization_std']],
                    visible=True
                )
            ))

        fig.update_layout(
            title='Average Budget Utilization',
            xaxis_title='Models',
            yaxis_title='Budget Used (M)',
            showlegend=False
        )

        return fig

    def plot_player_selection_frequency(self, metrics_list, model_names):
        fig = make_subplots(rows=len(metrics_list), cols=1,
                            subplot_titles=model_names,
                            vertical_spacing=0.05)

        for i, metrics in enumerate(metrics_list):
            players, counts = zip(*metrics['top_10_players'])
            fig.add_trace(
                go.Bar(x=players, y=counts,
                       marker_color=self.colors[i % len(self.colors)]),
                row=i + 1, col=1
            )

        fig.update_layout(
            title='Top 10 Most Selected Players by Model',
            showlegend=False,
            height=300 * len(metrics_list)
        )

        return fig

    def plot_round_performance(self, metrics_list, model_names):
        fig = go.Figure()

        for i, metrics in enumerate(metrics_list):
            rounds = list(metrics['round_performance'].keys())
            performances = list(metrics['round_performance'].values())
            fig.add_trace(go.Scatter(
                x=rounds,
                y=performances,
                mode='lines+markers',
                name=model_names[i],
                line=dict(color=self.colors[i % len(self.colors)])
            ))

        fig.update_layout(
            title='Performance by Round',
            xaxis_title='Round',
            yaxis_title='Performance'
        )

        return fig

    def plot_time_series(self, time_series_data_list, model_names):
        fig = go.Figure()

        for i, df in enumerate(time_series_data_list):
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['rolling_avg'],
                mode='lines',
                name=f"{model_names[i]} (7-day rolling avg)",
                line=dict(color=self.colors[i % len(self.colors)])
            ))

        fig.update_layout(
            title='Model Performance Over Time',
            xaxis_title='Date',
            yaxis_title='Performance (7-day rolling average)'
        )

        return fig

    def save_plots(self, figs, output_dir):
        for name, fig in figs.items():
            fig.write_html(f"{output_dir}/{name}.html")
            fig.write_image(f"{output_dir}/{name}.png")

    def visualize_all(self, metrics_list, time_series_data_list, model_names, output_dir):
        figs = {
            'metrics_comparison': self.plot_metrics_comparison(metrics_list, model_names),
            'budget_utilization': self.plot_budget_utilization(metrics_list, model_names),
            'player_selection_frequency': self.plot_player_selection_frequency(metrics_list, model_names),
            'round_performance': self.plot_round_performance(metrics_list, model_names),
            'time_series': self.plot_time_series(time_series_data_list, model_names)
        }

        self.save_plots(figs, output_dir)
        return figs