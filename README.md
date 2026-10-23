# 🎨 Scribblytics
### Where Every Scribble Tells a Story

Scribblytics is an interactive drawing analysis application that extracts visual features from drawings and groups them using Machine Learning. It helps users explore drawing measurements through an interactive dashboard.

## ✨ Features

- **Drawing Canvas:** Draw pictures directly in the application.
- **Ink Coverage Analysis:** Measures the percentage of the canvas occupied by the drawing.
- **Contour Detection:** Counts the external contours detected in a drawing.
- **Symmetry Analysis:** Measures the similarity between the left and right halves of a drawing.
- **Interactive Dashboard:** Displays drawing statistics and visual comparisons.
- **K-Means Clustering:** Groups drawings based on similarities in their numerical features.
- **CSV Support:** Upload previous datasets and download drawing measurements.
- **Drawing History:** View and compare drawings analyzed during the current session.

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Plotly

## ⚙️ Installation and Setup

1. Clone this repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Open the project folder:

   ```bash
   cd Scribblytics
   ```

3. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   ```

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

4. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

## 🧠 Machine Learning Approach

Scribblytics uses K-Means, an unsupervised machine learning algorithm, to group drawings according to their measured features.

The application extracts ink coverage, contour count, and symmetry score. These features are scaled before clustering so that differences in their numerical ranges do not unfairly dominate the comparison.

**Note:** The application groups drawings by numerical feature similarity. It does not identify the actual objects drawn.

## 📊 Dataset

Drawing measurements can be exported as a CSV file and uploaded later for further analysis. The CSV stores numerical features and drawing names, not the original drawing images.

## 🚀 Future Improvements

- Save drawing images alongside their measurements.
- Add persistent drawing history across sessions.
- Explore more image features and clustering techniques.
- Improve the dashboard with additional visualizations.

## 👩‍💻 Author

**Haasini Obilisetty**

AI & Data Science Undergraduate

## 📄 License

This project is available for learning and educational purposes.
