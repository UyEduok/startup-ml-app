import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_feature_distribution(df):
    st.subheader("Feature Distribution Pattern")

    sns.set_style("whitegrid")

    # Selected columns
    categorical_cols = ['state_code', 'city', 'category_code']

    numerical_cols = [
        'latitude', 'longitude',
        'age_first_funding_year', 'age_last_funding_year',
        'age_first_milestone_year', 'age_last_milestone_year',
        'relationships', 'funding_rounds',
        'funding_total_usd', 'milestones',
        'avg_participants'
    ]

    # Keep only columns that actually exist in the dataset
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    if not categorical_cols and not numerical_cols:
        st.warning("None of the selected feature columns were found in the dataset.")
        return


    # 1. Frequency Distribution (Categorical)
    with st.expander("1. Frequency Distribution for Categorical Features", expanded=True):
        if categorical_cols:
            for col in categorical_cols:
                st.markdown(f"### {col}")

                freq_table = df[col].value_counts(dropna=False).head(10).reset_index()
                freq_table.columns = [col, "Count"]
                st.dataframe(freq_table, use_container_width=True, hide_index=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                df[col].value_counts(dropna=False).head(10).plot(kind='bar', ax=ax)
                ax.set_title(f"Top Categories in {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.tick_params(axis='x', rotation=45)

                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No selected categorical columns were found in the dataset.")

 
    # 2. Distribution Plots (Numerical)
    with st.expander("2. Distribution Plots for Numerical Features", expanded=False):
        if numerical_cols:
            for col in numerical_cols:
                st.markdown(f"### {col}")

                fig, ax = plt.subplots(figsize=(6, 3))
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")

                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No selected numerical columns were found in the dataset.")


    # 3. Outlier Detection (IQR + Boxplot)
    with st.expander("3. Outlier Detection", expanded=False):
        if numerical_cols:
            outlier_summary = []

            for col in numerical_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()

                outlier_summary.append({
                    'Feature': col,
                    'Outlier_Count': outliers
                })

                st.markdown(f"### {col}")

                fig, ax = plt.subplots(figsize=(6, 2))
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Boxplot: {col}")

                st.pyplot(fig)
                plt.close(fig)

            outlier_df = pd.DataFrame(outlier_summary).sort_values(
                by='Outlier_Count', ascending=False
            )

            st.markdown("### Outlier Summary")
            st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        else:
            st.info("No selected numerical columns were found in the dataset.")


    # 4. Skewness Check
    with st.expander("4. Skewness Check", expanded=False):
        if numerical_cols:
            skewness_df = pd.DataFrame({
                'Feature': numerical_cols,
                'Skewness': [df[col].dropna().skew() for col in numerical_cols]
            }).sort_values(by='Skewness', key=lambda x: x.abs(), ascending=False)

            st.dataframe(skewness_df, use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(skewness_df['Feature'], skewness_df['Skewness'])
            ax.axhline(0, linestyle='--')
            ax.set_title("Skewness of Numerical Features")
            ax.tick_params(axis='x', rotation=45)

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No selected numerical columns were found in the dataset.")