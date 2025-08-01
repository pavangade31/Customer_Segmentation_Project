# Shopper Spectrum: Customer Segmentation & Product Recommendations

![Shopper Spectrum](https://img.shields.io/badge/Shopper%20Spectrum-Streamlit%20App-blue.svg)  
[![Releases](https://img.shields.io/badge/Releases-Check%20Here-brightgreen.svg)](https://github.com/zigibir/Shopper-Spectrum_-Segmentation-and-Recomm/releases)

## Overview

Shopper Spectrum is a Streamlit application designed for customer segmentation and product recommendations using e-commerce data. This project employs RFM (Recency, Frequency, Monetary) analysis and KMeans clustering for customer segmentation. It also integrates collaborative filtering techniques for generating product recommendations.

## Features

- **Customer Segmentation**: Use RFM analysis combined with KMeans clustering to identify distinct customer segments.
- **Product Recommendations**: Leverage collaborative filtering to suggest products based on user preferences and behavior.
- **Data Visualization**: Interactive charts and graphs for better understanding of customer behavior and product trends.
- **Real-Time Prediction**: Get immediate recommendations based on user input.

## Technologies Used

- **Python Libraries**:
  - `numpy`: For numerical operations.
  - `pandas`: For data manipulation and analysis.
  - `scikit-learn`: For machine learning algorithms including KMeans clustering.
  - `streamlit`: For building the web application interface.
- **Data Processing**:
  - Data cleaning and transformation techniques for preparing e-commerce data.
  - Exploratory Data Analysis (EDA) for insights into customer behavior.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zigibir/Shopper-Spectrum_-Segmentation-and-Recomm.git
   cd Shopper-Spectrum_-Segmentation-and-Recomm
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Access the app**: Open your web browser and go to `http://localhost:8501`.

## Usage

### Customer Segmentation

1. **Input Data**: Upload your e-commerce data in CSV format.
2. **Select Segmentation Parameters**: Choose the RFM metrics to segment customers.
3. **View Results**: The app will display customer segments with visualizations.

### Product Recommendations

1. **Input User Data**: Provide user preferences or past purchase history.
2. **Get Recommendations**: The app will generate a list of recommended products based on collaborative filtering.

## Data Cleaning & Transformation

Data cleaning is essential for accurate analysis. This project includes:

- Handling missing values.
- Removing duplicates.
- Normalizing data for consistent analysis.

## Exploratory Data Analysis (EDA)

The EDA phase helps uncover patterns in the data. Key techniques used:

- **Pivot Tables**: To summarize data.
- **Visualizations**: Charts and graphs to illustrate customer behavior.

## Feature Engineering

Feature engineering enhances model performance. Techniques used include:

- Creating new features based on existing data.
- Transforming categorical data into numerical formats.

## KMeans Clustering

KMeans clustering is used for segmenting customers. The process involves:

1. **Choosing the Number of Clusters**: Using the elbow method to determine optimal clusters.
2. **Fitting the Model**: Applying KMeans to the RFM data.
3. **Interpreting Results**: Analyzing cluster centers to understand customer segments.

## Collaborative Filtering

Collaborative filtering generates product recommendations. Key steps include:

- **User-Item Matrix**: Constructing a matrix of user preferences.
- **Cosine Similarity**: Calculating similarity between users or items.
- **Generating Recommendations**: Using the similarity scores to suggest products.

## Real-Time Prediction

The app provides real-time predictions based on user input. This feature enhances user engagement by offering personalized recommendations.

## Data Visualization

Data visualization plays a crucial role in understanding the data. The app includes:

- **Interactive Charts**: For exploring customer segments.
- **Graphs**: To visualize product recommendations.

## Screenshots

![Customer Segmentation](https://via.placeholder.com/800x400.png?text=Customer+Segmentation)  
![Product Recommendations](https://via.placeholder.com/800x400.png?text=Product+Recommendations)

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch to your forked repository.
5. Open a pull request to the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out via the GitHub Issues section or contact the repository owner.

## Releases

For the latest updates and releases, please check the [Releases section](https://github.com/zigibir/Shopper-Spectrum_-Segmentation-and-Recomm/releases). Download the latest version and execute the app to explore its features.

## Topics

This project covers a range of topics, including:

- Collaborative Filtering
- Cosine Similarity
- Customer Segmentation
- Data Cleaning
- Data Transformation
- Data Visualization
- Exploratory Data Analysis (EDA)
- Feature Engineering
- KMeans Clustering
- Machine Learning
- NumPy
- Pandas
- Pivot Tables
- Product Recommendation
- Real-Time Prediction
- RFM Analysis
- Scikit-learn
- Standard Scaler
- Streamlit App

Explore these topics to deepen your understanding of customer segmentation and product recommendations in e-commerce.

## Acknowledgments

Thank you to all the contributors and the open-source community for providing valuable resources and libraries that made this project possible. 

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)

For further information, please refer to the official documentation of the libraries used in this project.
