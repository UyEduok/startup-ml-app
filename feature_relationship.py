import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import math


def show_feature_relationship(df):
    st.subheader("Feature Relationship & Correlation Analysis")

    sns.set_style("whitegrid")

    # Columns
    numerical_cols = [
        'latitude', 'longitude',
        'age_first_funding_year', 'age_last_funding_year',
        'age_first_milestone_year', 'age_last_milestone_year',
        'relationships', 'funding_rounds',
        'funding_total_usd', 'milestones',
        'avg_participants'
    ]

    categorical_cols = ['state_code', 'city', 'category_code']

    # Keep only existing columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    if not numerical_cols:
        st.warning("No numerical columns found for correlation analysis.")
        return


    # 1. Correlation Heatmap
    with st.expander("1. Correlation Heatmap", expanded=True):

        corr_matrix = df[numerical_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            square=True,
            ax=ax
        )

        ax.set_title("Correlation Heatmap of Numerical Features")

        st.pyplot(fig)
        plt.close(fig)


    # 2. Numerical vs Target
    with st.expander("2. Numerical Features vs Target", expanded=False):

        if 'status' not in df.columns:
            st.warning("Target column 'status' not found.")
        else:
            n_cols = 3
            n_rows = math.ceil(len(numerical_cols) / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4.5))
            axes = axes.flatten()

            for i, col in enumerate(numerical_cols):
                sns.stripplot(
                    data=df,
                    x='status',
                    y=col,
                    jitter=0.25,
                    alpha=0.6,
                    ax=axes[i]
                )
                axes[i].set_title(f"{col} vs Status", fontsize=10)
                axes[i].set_xlabel("Status")
                axes[i].set_ylabel(col)

            # Remove empty plots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle(
                "Relationship Between Numerical Features and Target",
                fontsize=14
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


    # 3. Categorical vs Target
    with st.expander("3. Categorical Features vs Target", expanded=False):

        if 'status' not in df.columns:
            st.warning("Target column 'status' not found.")
        elif not categorical_cols:
            st.info("No categorical columns found.")
        else:
            fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(14, 6 * len(categorical_cols)))

            if len(categorical_cols) == 1:
                axes = [axes]

            for i, col in enumerate(categorical_cols):
                top_categories = df[col].value_counts().head(10).index
                filtered_df = df[df[col].isin(top_categories)]

                sns.countplot(
                    data=filtered_df,
                    x=col,
                    hue='status',
                    ax=axes[i]
                )

                axes[i].set_title(f"{col} vs Status")
                axes[i].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)