import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import numpy as np

st.set_page_config(
    page_title="KYC Workshop",
    page_icon=":policeman:",
    layout="wide",
)

st.title("Aprende a medir el éxito y riesgo de tus procesos KYC 🏦👮🏻")


def get_y_pred(fraud_threshold):
    global data
    y_pred = data.apply(
        lambda row: 1
        if row.FRAUD_SCORE >= fraud_threshold
        else 0,
        axis=1,
    )

    return y_pred


data = pd.read_csv("./data.csv")
thresholds = pd.read_csv("./thresholds.csv")
y_true = data.apply(
    lambda row: 1 if row.FRAUD == "YES" else 0, axis=1
)

data.FRAUD_SCORE = data.FRAUD_SCORE.map(lambda x: int(x))

st.dataframe(data)
# add download button
st.download_button(
    label="Descargar datos",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="data_transformed.csv",
    mime="text/csv",
)




st.subheader("Matriz de confusión")

# create an empty editable table of 2x2
col_1, col_2 = st.columns(2)
with col_1:
    tn_input = st.text_input("Verdaderos Negativos", value=None, key="tn")
    fn_input = st.text_input("Falsos Negativos", value=None, key="fn")

with col_2:
    fp_input = st.text_input("Falsos Positivos", value=None, key="fp")
    tp_input = st.text_input("Verdaderos Positivos", value=None, key="tp")

if st.button("Validar respuesta"):
    if (
        tn_input == str(8062)
        and fn_input == str(3)
        and fp_input == str(111)
        and tp_input == str(300)
    ):
        st.success("Respuesta correcta")
        st.balloons()
    else:
        st.error("Respuesta incorrecta")

if st.checkbox("Mostrar matriz de confusión"):
    # build a confusion matrix display with plotly heatmap
    st.table(
        {
            "Verdaderos Negativos": [8062],
            "Falsos Negativos": [3],
            "Falsos Positivos": [111],
            "Verdaderos Positivos": [300],
        }
    )


# plot FAR and FRR for each threshold
st.subheader("Efecto de los umbrales de riesgo")


fraud_threshold = st.slider(
    "Umbral de fraude",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
)

y_pred = get_y_pred(fraud_threshold)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])


tn, fp, fn, tp = cm.ravel()

far = fn / (tp + fn) if (tp + fn) > 0 else 0
frr = fp / (tn + fp) if (tn + fp) > 0 else 0

col_far, col_frr = st.columns(2)

col_far.metric("False Acceptance Rate", f"{far:.2%}")
col_frr.metric("False Rejection Rate", f"{frr:.2%}")
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=thresholds["THRESHOLD"],
        y=thresholds["FAR"],
        mode="lines",
        name="False Acceptance Rate",
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=thresholds["THRESHOLD"],
        y=thresholds["FRR"],
        mode="lines",
        name="False Rejection Rate",
    ),
    secondary_y=True,
)

fig.update_yaxes(title_text="False Acceptance Rate", secondary_y=False)
fig.update_yaxes(title_text="False Rejection Rate", secondary_y=True)

fig.update_layout(
    title="FAR and FRR for face match at different similarity thresholds",
    xaxis_title="Threshold",
)

# format x ticks from 0 to 100 with 5 increases and as percentage
fig.update_xaxes(
    tickvals=np.arange(0, 105, 5) / 100,
    ticktext=[f"{x:.0%}" for x in np.arange(0, 105, 5) / 100],
)

# format both y axes to show percentage
fig.update_yaxes(
    tickvals=np.arange(0, 105, 10) / 100,
    ticktext=[f"{x:.0%}" for x in np.arange(0, 105, 10) / 100],
)

# format hover template with percentages include the threshold and value
fig.update_traces(
    hovertemplate="Threshold: %{x:.0%}<br>Value: %{y:.1%}"
)

# locate legends below the title
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    )
)

# adjust plot width and height for a slide
fig.update_layout(
    width=1200,
    height=800,
)

# draw vertical line in fraud_threshold with the text "Umbral de fraude"
fig.add_shape(
    dict(
        type="line",
        x0=fraud_threshold,
        y0=0,
        x1=fraud_threshold,
        y1=1,
        line=dict(color="red", width=2),
        layer="below",
        xref="x",
        yref="paper",
        opacity=0.5,
    )
)

if st.checkbox("Mostrar curva FAR y FRR"):
    st.plotly_chart(fig, use_container_width=True)
    st.table(
        {
            "Verdaderos Negativos": [tn],
            "Falsos Negativos": [fn],
            "Falsos Positivos": [fp],
            "Verdaderos Positivos": [tp],
        }
    )