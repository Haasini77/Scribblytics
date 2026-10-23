
import hashlib

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from streamlit_drawable_canvas import st_canvas


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Scribblytics",
    page_icon="🎨",
    layout="wide",
)


# ---------------- SESSION STATE ----------------

if "drawing_history" not in st.session_state:
    st.session_state.drawing_history = []

if "uploaded_history" not in st.session_state:
    st.session_state.uploaded_history = pd.DataFrame()

if "last_uploaded_hash" not in st.session_state:
    st.session_state.last_uploaded_hash = None


# ---------------- CONSTANTS ----------------

required_columns = [
    "Drawing Name",
    "Ink Coverage (%)",
    "Contour Count",
    "Symmetry Score (%)",
]

feature_cols = [
    "Ink Coverage (%)",
    "Contour Count",
    "Symmetry Score (%)",
]


# ---------------- HEADER ----------------

st.title("🎨 Scribblytics")
st.subheader("Where Every Scribble Tells a Story.")

st.write(
    "Draw doodles, analyse their visual features, "
    "and discover groups of similar drawings using Machine Learning."
)

st.info(
    "Scribblytics analyses numerical drawing features. "
    "It does not automatically recognise objects such as flowers or houses."
)


# ---------------- DRAWING CANVAS ----------------

st.divider()
st.header("✏️ Draw Your Scribble")

drawing_name = st.text_input(
    "Drawing name",
    placeholder="e.g. Flower, Apple, House",
)

drawing = st_canvas(
    fill_color="rgba(255, 255, 255, 0.1)",
    stroke_width=4,
    stroke_color="#7C3AED",
    background_color="#FFFFFF",
    height=350,
    width=600,
    drawing_mode="freedraw",
    key="scribblytics_canvas",
    return_image_data=True,
    update_streamlit=True,
)


# ---------------- ANALYSE AND SAVE DRAWING ----------------

if st.button("🔍 Analyze and Save Drawing", type="primary"):

    if drawing.image_data is None:
        st.warning("Please draw something first!")

    else:
        image = drawing.image_data.astype(np.uint8)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2GRAY,
        )

        _, binary = cv2.threshold(
            gray,
            245,
            255,
            cv2.THRESH_BINARY_INV,
        )

        # Feature 1: Ink coverage

        ink_pixels = np.count_nonzero(binary)

        ink_coverage = (
            ink_pixels / binary.size
        ) * 100

        if ink_pixels == 0:
            st.warning(
                "The canvas is blank. Please draw something before analysing."
            )

        else:
            # Feature 2: Contour count

            contours, _ = cv2.findContours(
                binary.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            contour_count = len(contours)

            # Feature 3: Left-right pixel-overlap symmetry

            height, width = binary.shape
            mid = width // 2

            left = binary[:, :mid] > 0
            right = np.fliplr(
                binary[:, width - mid:] > 0
            )

            intersection = np.count_nonzero(
                left & right
            )

            union = np.count_nonzero(
                left | right
            )

            symmetry_score = (
                intersection / union * 100
                if union > 0
                else 0.0
            )

            # Save drawing measurements

            record = {
                "Drawing ID": (
                    len(st.session_state.drawing_history) + 1
                ),
                "Drawing Name": (
                    drawing_name.strip() or "Untitled"
                ),
                "Ink Coverage (%)": round(
                    float(ink_coverage), 2
                ),
                "Contour Count": int(contour_count),
                "Symmetry Score (%)": round(
                    float(symmetry_score), 2
                ),
            }

            st.session_state.drawing_history.append(
                record
            )

            st.success(
                f"Drawing '{record['Drawing Name']}' analysed and saved!"
            )

            # Display current drawing measurements

            st.subheader("📊 Current Drawing Analysis")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Ink Coverage",
                f"{ink_coverage:.2f}%",
            )

            c2.metric(
                "Detected Contours",
                contour_count,
            )

            c3.metric(
                "Symmetry Score",
                f"{symmetry_score:.2f}%",
            )

            # Display processed drawing

            st.subheader("🖼️ Processed Drawing")

            st.image(
                binary,
                caption="White strokes on a black background",
                clamp=True,
                width=600,
            )

            # Draw detected boundaries

            outlined = cv2.cvtColor(
                binary,
                cv2.COLOR_GRAY2RGB,
            )

            cv2.drawContours(
                outlined,
                contours,
                -1,
                (0, 255, 0),
                2,
            )

            st.subheader("🟢 Detected Drawing Outlines")

            st.image(
                outlined,
                caption="Green lines show detected boundaries",
                width=600,
            )


# ---------------- LOAD A PREVIOUS DATASET ----------------

st.divider()
st.subheader("📂 Load a Previous Dataset")

st.write(
    "Upload a CSV file previously downloaded from Scribblytics "
    "to restore its recorded drawing measurements."
)

uploaded_file = st.file_uploader(
    "Upload your Scribblytics CSV file",
    type=["csv"],
    key="history_csv_uploader",
)

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    upload_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    if upload_hash != st.session_state.last_uploaded_hash:

        try:
            loaded_df = pd.read_csv(
                uploaded_file
            )

            if all(
                col in loaded_df.columns
                for col in required_columns
            ):

                # Validate numerical feature columns

                for col in feature_cols:
                    loaded_df[col] = pd.to_numeric(
                        loaded_df[col],
                        errors="coerce",
                    )

                loaded_df = loaded_df.dropna(
                    subset=required_columns
                ).copy()

                # Keep valid measurements only

                loaded_df = loaded_df[
                    (loaded_df["Ink Coverage (%)"] >= 0)
                    & (loaded_df["Ink Coverage (%)"] <= 100)
                    & (loaded_df["Contour Count"] >= 0)
                    & (loaded_df["Symmetry Score (%)"] >= 0)
                    & (loaded_df["Symmetry Score (%)"] <= 100)
                ].copy()

                if not loaded_df.empty:

                    st.session_state.uploaded_history = (
                        loaded_df.copy()
                    )

                    st.session_state.last_uploaded_hash = (
                        upload_hash
                    )

                    st.success(
                        f"Dataset loaded successfully! "
                        f"{len(loaded_df)} valid records found."
                    )

                    st.rerun()

                else:
                    st.session_state.uploaded_history = (
                        pd.DataFrame()
                    )

                    st.session_state.last_uploaded_hash = (
                        upload_hash
                    )

                    st.error(
                        "The CSV contains no valid drawing records."
                    )

            else:
                st.error(
                    "This CSV is missing required columns: "
                    + ", ".join(required_columns)
                )

        except Exception as exc:
            st.error(
                f"Unable to read the CSV file: {exc}"
            )


# ---------------- PREPARE DATASET ----------------

session_df = pd.DataFrame(
    st.session_state.drawing_history
)

uploaded_df = (
    st.session_state.uploaded_history.copy()
)

if not session_df.empty and not uploaded_df.empty:

    data_df = pd.concat(
        [uploaded_df, session_df],
        ignore_index=True,
    )

elif not session_df.empty:

    data_df = session_df.copy()

else:

    data_df = uploaded_df.copy()


# ---------------- CLEAN DATASET ----------------

if (
    not data_df.empty
    and all(
        col in data_df.columns
        for col in required_columns
    )
):

    data_df = data_df.dropna(
        subset=required_columns
    ).copy()

    for col in feature_cols:
        data_df[col] = pd.to_numeric(
            data_df[col],
            errors="coerce",
        )

    data_df = data_df.dropna(
        subset=feature_cols
    ).copy()

    # Keep measurements within sensible ranges

    data_df = data_df[
        (data_df["Ink Coverage (%)"] >= 0)
        & (data_df["Ink Coverage (%)"] <= 100)
        & (data_df["Contour Count"] >= 0)
        & (data_df["Symmetry Score (%)"] >= 0)
        & (data_df["Symmetry Score (%)"] <= 100)
    ].copy()

    # Remove exact duplicate records

    data_df = data_df.drop_duplicates(
        subset=[
            "Drawing Name",
            "Ink Coverage (%)",
            "Contour Count",
            "Symmetry Score (%)",
        ]
    ).reset_index(drop=True)

    # Assign IDs for the displayed dataset

    data_df["Drawing ID"] = range(
        1,
        len(data_df) + 1,
    )

else:

    data_df = pd.DataFrame(
        columns=[
            "Drawing ID",
            *required_columns,
        ]
    )


# ---------------- DRAWING DASHBOARD ----------------

st.divider()
st.header("📊 Drawing Dashboard")

if not data_df.empty:

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Drawings",
        len(data_df),
    )

    c2.metric(
        "Average Ink Coverage",
        f"{data_df['Ink Coverage (%)'].mean():.2f}%",
    )

    c3.metric(
        "Average Contours",
        f"{data_df['Contour Count'].mean():.2f}",
    )

    c4.metric(
        "Average Symmetry",
        f"{data_df['Symmetry Score (%)'].mean():.2f}%",
    )

    # Feature comparison chart

    st.subheader("📈 Feature Comparison")

    feature_choice = st.selectbox(
        "Choose a feature to compare",
        feature_cols,
    )

    bar_fig = px.bar(
        data_df,
        x="Drawing Name",
        y=feature_choice,
        color="Drawing Name",
        title=f"{feature_choice} by Drawing",
        hover_data=["Drawing ID"],
    )

    st.plotly_chart(
        bar_fig,
        use_container_width=True,
    )

else:

    st.info(
        "Save some drawings or upload a dataset "
        "to see your dashboard."
    )


# ---------------- DRAWING HISTORY ----------------

st.divider()
st.header("📚 Drawing History")

if not data_df.empty:

    history_display = data_df[
        [
            "Drawing ID",
            "Drawing Name",
            "Ink Coverage (%)",
            "Contour Count",
            "Symmetry Score (%)",
        ]
    ].copy()

    st.dataframe(
        history_display,
        hide_index=True,
        use_container_width=True,
    )

    st.write(
        f"**Total drawings: {len(history_display)}**"
    )

    # Download measurements as CSV

    csv_data = history_display.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Dataset as CSV",
        data=csv_data,
        file_name="scribblytics_dataset.csv",
        mime="text/csv",
    )

else:

    st.info(
        "No drawings available yet. Draw something and click "
        "'Analyze and Save Drawing' to begin, or upload a CSV above."
    )


# ---------------- CLEAR CURRENT SESSION DRAWINGS ----------------

if st.button("🗑️ Clear Current Session Drawings"):

    st.session_state.drawing_history = []

    st.rerun()


# ---------------- MACHINE LEARNING: K-MEANS ----------------

st.divider()
st.header("🤖 Machine Learning: Drawing Clusters")

st.write(
    "K-Means groups drawings using ink coverage, contour count, "
    "and symmetry score. It does not identify the object."
)

if len(data_df) >= 3:

    # Keep the slider minimum below its maximum

    max_clusters = min(
        5,
        len(data_df),
    )

    default_clusters = min(
        3,
        max_clusters,
    )

    n_clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=max_clusters,
        value=default_clusters,
    )

    # Prepare and scale numerical features

    features = data_df[
        feature_cols
    ].astype(float)

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    # Train K-Means

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    clustered_df = data_df.copy()

    clustered_df["Cluster"] = (
        model.fit_predict(scaled_features) + 1
    )

    # Show cluster assignments

    st.subheader("📋 Cluster Assignments")

    st.dataframe(
        clustered_df[
            [
                "Drawing Name",
                "Ink Coverage (%)",
                "Contour Count",
                "Symmetry Score (%)",
                "Cluster",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    # Chart: number of drawings in each cluster

    cluster_counts = (
        clustered_df["Cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Number of Drawings")
    )

    st.subheader("📦 Drawings per Cluster")

    count_fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Number of Drawings",
        color="Cluster",
        title="Number of Drawings in Each Cluster",
    )

    st.plotly_chart(
        count_fig,
        use_container_width=True,
    )

    # Chart: compare drawing features

    st.subheader(
        "🎨 Drawing Groups Based on Visual Features"
    )

    fig = px.scatter(
        clustered_df,
        x="Ink Coverage (%)",
        y="Symmetry Score (%)",
        color=clustered_df["Cluster"].astype(str),
        size="Contour Count",
        size_max=35,
        hover_name="Drawing Name",
        hover_data=[
            "Ink Coverage (%)",
            "Contour Count",
            "Symmetry Score (%)",
            "Cluster",
        ],
        title="Drawing Groups Based on Visual Features",
        labels={
            "color": "Cluster",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Each colour represents a cluster. Dot size represents "
        "contour count. Cluster numbers are labels, not rankings."
    )

    st.caption(
        "The groups depend on the measured features and "
        "the selected number of clusters."
    )

elif len(data_df) > 0:

    st.info(
        "Save or load at least 3 drawings to run K-Means clustering."
    )

else:

    st.info(
        "Save some drawings or load a dataset first. "
        "The ML section requires at least 3 valid drawings."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Scribblytics measures drawing features and groups drawings "
    "using Machine Learning. Symmetry is a basic pixel-overlap "
    "measurement, not a measure of artistic quality."
)

st.caption(
    "Drawing history is held in the current Streamlit session. "
    "Download the CSV to keep a copy for future use."
)
