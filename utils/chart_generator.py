import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime, timedelta

class ChartGenerator:
    """Generates interactive charts for financial data visualization"""

    def __init__(self):
        # Color scheme for categories
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
            '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
            '#EE5A24', '#009432', '#0652DD', '#9980FA', '#FFC312'
        ]

    def create_category_pie_chart(self, df: pd.DataFrame):
        """Create a pie chart showing both income and expenses by category"""
        try:
            if df.empty:
                return None

            # Separate income and expense transactions
            income_df = df[df['amount'] > 0].copy()
            expense_df = df[df['amount'] < 0].copy()

            # Group by category and sum amounts
            income_totals = income_df.groupby('category')['amount'].sum() if not income_df.empty else pd.Series()
            expense_totals = expense_df.groupby('category')['amount'].sum().abs() if not expense_df.empty else pd.Series()

            # Combine all categories
            all_categories = set()
            if not income_totals.empty:
                all_categories.update(income_totals.index)
            if not expense_totals.empty:
                all_categories.update(expense_totals.index)

            if not all_categories:
                return None

            # Create combined data
            category_data = []
            labels = []
            colors = []
            
            for category in sorted(all_categories):
                income_amount = income_totals.get(category, 0)
                expense_amount = expense_totals.get(category, 0)
                
                if income_amount > 0:
                    category_data.append(income_amount)
                    labels.append(f"{category} (Income)")
                    colors.append('#2ECC71')  # Green for income
                
                if expense_amount > 0:
                    category_data.append(expense_amount)
                    labels.append(f"{category} (Expense)")
                    colors.append('#E74C3C')  # Red for expenses

            if not category_data:
                return None

            # Create pie chart
            fig = px.pie(
                values=category_data,
                names=labels,
                title="Income & Spending Distribution by Category",
                color_discrete_sequence=colors if len(colors) == len(category_data) else px.colors.qualitative.Set3
            )

            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
            )

            fig.update_layout(
                showlegend=True,
                height=400
            )

            return fig

        except Exception as e:
            print(f"Error creating category pie chart: {str(e)}")
            return None

    def create_monthly_trend_chart(self, df):
        """Create monthly trend chart"""
        try:
            if df.empty:
                return None

            df_copy = df.copy()
            # Handle various date formats more flexibly
            df_copy['date'] = pd.to_datetime(df_copy['date'], format='mixed', dayfirst=True, errors='coerce')

            df_copy['month_year'] = df_copy['date'].dt.to_period('M')

            # Separate income and expenses
            income_df = df_copy[df_copy['amount'] > 0]
            expense_df = df_copy[df_copy['amount'] < 0]

            # Group by month
            monthly_income = income_df.groupby('month_year')['amount'].sum()
            monthly_expenses = expense_df.groupby('month_year')['amount'].sum().abs()

            # Create date range for all months
            if not monthly_income.empty or not monthly_expenses.empty:
                all_months = pd.period_range(
                    start=min(monthly_income.index.min() if not monthly_income.empty else monthly_expenses.index.min(),
                             monthly_expenses.index.min() if not monthly_expenses.empty else monthly_income.index.min()),
                    end=max(monthly_income.index.max() if not monthly_income.empty else monthly_expenses.index.max(),
                           monthly_expenses.index.max() if not monthly_expenses.empty else monthly_income.index.max()),
                    freq='M'
                )

                # Reindex to include all months
                monthly_income = monthly_income.reindex(all_months, fill_value=0)
                monthly_expenses = monthly_expenses.reindex(all_months, fill_value=0)

                # Convert period index to string for plotting
                months_str = [str(month) for month in all_months]

                # Create line chart
                fig = go.Figure()

                # Add income line
                fig.add_trace(go.Scatter(
                    x=months_str,
                    y=monthly_income.values,
                    mode='lines+markers',
                    name='Income',
                    line=dict(color='#2ECC71', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>Income</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Amount: ₹%{y:,.2f}<br>' +
                                 '<extra></extra>'
                ))

                # Add expenses line
                fig.add_trace(go.Scatter(
                    x=months_str,
                    y=monthly_expenses.values,
                    mode='lines+markers',
                    name='Expenses',
                    line=dict(color='#E74C3C', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>Expenses</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Amount: ₹%{y:,.2f}<br>' +
                                 '<extra></extra>'
                ))

                # Add net savings line
                net_savings = monthly_income.values - monthly_expenses.values
                fig.add_trace(go.Scatter(
                    x=months_str,
                    y=net_savings,
                    mode='lines+markers',
                    name='Net Savings',
                    line=dict(color='#3498DB', width=2, dash='dash'),
                    marker=dict(size=6),
                    hovertemplate='<b>Net Savings</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Amount: ₹%{y:,.2f}<br>' +
                                 '<extra></extra>'
                ))

                # Customize layout
                fig.update_layout(
                    title='Monthly Financial Trend',
                    xaxis_title='Month',
                    yaxis_title='Amount (₹)',
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )

                # Format y-axis to show currency
                fig.update_layout(yaxis=dict(tickformat='₹,.0f'))

                return fig

            return None
        except Exception as e:
            print(f"Error in create_monthly_trend_chart: {e}")
            return None

    def create_category_trend_chart(self, df: pd.DataFrame, category: str) -> Optional[go.Figure]:
        """
        Create a trend chart for a specific category

        Args:
            df: DataFrame with transaction data
            category: Category to show trend for

        Returns:
            Plotly figure object or None if no data
        """
        if df.empty or 'category' not in df.columns:
            return None

        # Filter by category
        category_df = df[df['category'] == category].copy()

        if category_df.empty:
            return None

        # Convert date and group by month
        category_df['date'] = pd.to_datetime(category_df['date'])
        category_df['month_year'] = category_df['date'].dt.to_period('M')

        monthly_amounts = category_df.groupby('month_year')['amount'].sum()

        # Convert to absolute values for expenses
        if monthly_amounts.mean() < 0:
            monthly_amounts = monthly_amounts.abs()

        # Create line chart
        months_str = [str(month) for month in monthly_amounts.index]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=months_str,
            y=monthly_amounts.values,
            mode='lines+markers',
            name=category,
            line=dict(width=3),
            marker=dict(size=8),
            fill='tonexty' if len(monthly_amounts) > 1 else None,
            hovertemplate=f'<b>{category}</b><br>' +
                         'Month: %{x}<br>' +
                         'Amount: ₹%{y:,.2f}<br>' +
                         '<extra></extra>'
        ))

        fig.update_layout(
            title=f'{category} - Monthly Trend',
            xaxis_title='Month',
            yaxis_title='Amount (₹)',
            hovermode='x'
        )

        fig.update_layout(yaxis=dict(tickformat='₹,.0f'))

        return fig

    def create_daily_spending_chart(self, df: pd.DataFrame, days: int = 30) -> Optional[go.Figure]:
        """
        Create a bar chart showing daily spending for the last N days

        Args:
            df: DataFrame with transaction data
            days: Number of recent days to show

        Returns:
            Plotly figure object or None if no data
        """
        if df.empty:
            return None

        # Get recent transactions
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])

        # Filter to last N days
        end_date = df_copy['date'].max()
        start_date = end_date - timedelta(days=days)
        recent_df = df_copy[df_copy['date'] >= start_date]

        if recent_df.empty:
            return None

        # Get daily spending (expenses only)
        expense_df = recent_df[recent_df['amount'] < 0]
        daily_spending = expense_df.groupby('date')['amount'].sum().abs()

        # Create date range for all days
        date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        daily_spending = daily_spending.reindex(date_range, fill_value=0)

        # Create bar chart
        fig = px.bar(
            x=daily_spending.index.strftime('%Y-%m-%d'),
            y=daily_spending.values,
            title=f'Daily Spending - Last {days} Days',
            labels={'x': 'Date', 'y': 'Amount (₹)'},
            color=daily_spending.values,
            color_continuous_scale='Reds'
        )

        fig.update_traces(
            hovertemplate='<b>Daily Spending</b><br>' +
                         'Date: %{x}<br>' +
                         'Amount: ₹%{y:,.2f}<br>' +
                         '<extra></extra>'
        )

        fig.update_layout(
            showlegend=False,
            xaxis_tickangle=-45
        )

        return fig

    def create_transaction_volume_chart(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Create a chart showing transaction volume over time

        Args:
            df: DataFrame with transaction data

        Returns:
            Plotly figure object or None if no data
        """
        if df.empty:
            return None

        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['month_year'] = df_copy['date'].dt.to_period('M')

        # Count transactions per month
        monthly_volume = df_copy.groupby('month_year').size()

        # Separate by transaction type
        monthly_income_volume = df_copy[df_copy['amount'] > 0].groupby('month_year').size()
        monthly_expense_volume = df_copy[df_copy['amount'] < 0].groupby('month_year').size()

        months_str = [str(month) for month in monthly_volume.index]

        # Create stacked bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=months_str,
            y=monthly_income_volume.reindex(monthly_volume.index, fill_value=0),
            name='Income Transactions',
            marker_color='#2ECC71',
            hovertemplate='<b>Income Transactions</b><br>' +
                         'Month: %{x}<br>' +
                         'Count: %{y}<br>' +
                         '<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            x=months_str,
            y=monthly_expense_volume.reindex(monthly_volume.index, fill_value=0),
            name='Expense Transactions',
            marker_color='#E74C3C',
            hovertemplate='<b>Expense Transactions</b><br>' +
                         'Month: %{x}<br>' +
                         'Count: %{y}<br>' +
                         '<extra></extra>'
        ))

        fig.update_layout(
            title='Transaction Volume by Month',
            xaxis_title='Month',
            yaxis_title='Number of Transactions',
            barmode='stack'
        )

        return fig

    def create_top_merchants_chart(self, df: pd.DataFrame, top_n: int = 10) -> Optional[go.Figure]:
        """
        Create a horizontal bar chart showing top merchants/payees

        Args:
            df: DataFrame with transaction data
            top_n: Number of top merchants to show

        Returns:
            Plotly figure object or None if no data
        """
        if df.empty:
            return None

        # Extract merchant names from narration (simplified)
        expense_df = df[df['amount'] < 0].copy()

        if expense_df.empty:
            return None

        # Group by narration and sum amounts
        merchant_spending = expense_df.groupby('narration')['amount'].sum().abs()
        top_merchants = merchant_spending.nlargest(top_n)

        # Create horizontal bar chart
        fig = px.bar(
            x=top_merchants.values,
            y=top_merchants.index,
            orientation='h',
            title=f'Top {top_n} Spending Categories/Merchants',
            labels={'x': 'Amount (₹)', 'y': 'Merchant/Description'},
            color=top_merchants.values,
            color_continuous_scale='Reds'
        )

        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>' +
                         'Amount: ₹%{x:,.2f}<br>' +
                         '<extra></extra>'
        )

        fig.update_layout(
            showlegend=False,
            height=max(400, top_n * 40)  # Adjust height based on number of items
        )

        return fig