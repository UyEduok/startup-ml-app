import streamlit as st
import matplotlib.pyplot as plt


def show_target_analysis(df):
    st.subheader("Target Variable Analysis")

    # 1. Distribution (Count)
    st.markdown("### 1. Target Distribution (Count)")

    status_count = df['status'].value_counts()

    st.dataframe(
        status_count.reset_index().rename(
            columns={"index": "Status", "status": "Count"}
        ),
        use_container_width=True,
        hide_index=True
    )


    # 2. Class Balance (Percentage)
    st.markdown("### 2. Target Distribution (Percentage)")

    status_percent = df['status'].value_counts(normalize=True) * 100

    st.dataframe(
        status_percent.reset_index().rename(
            columns={"index": "Status", "status": "Percentage (%)"}
        ),
        use_container_width=True,
        hide_index=True
    )

    # 3. Bar Chart
    st.markdown("### 3. Distribution Plot")

    fig, ax = plt.subplots(figsize=(6, 5))

    status_count.plot(kind='bar', ax=ax)

    ax.set_title("Distribution of Target Variable (Status)")
    ax.set_xlabel("Status")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=0)

    # annotate values on bars
    for i, v in enumerate(status_count):
        ax.text(i, v, str(v), ha='center', va='bottom')

    st.pyplot(fig)