Bengaluru PG Market Analysis

A machine learning project for analyzing and segmenting Bengaluru PG (Paying Guest) listings into meaningful market segments and predicting the segment of new PG listings.

📌 Project Overview

Bengaluru has a large PG and co-living market where listings differ in terms of rent, safety, commute convenience, ratings, and amenities. Comparing PGs using rent alone does not provide a complete picture.

This project uses a two-stage machine learning approach:

1. K-Means Clustering is used to segment PG listings into four business-oriented market segments.
2. Classification models are trained to predict the market segment of a new PG listing.

The final model is integrated into a Streamlit application for real-time prediction.

---

🎯 Objectives

- Clean and standardize a messy PG listings dataset.
- Perform exploratory data analysis to understand rent, safety, commute, and amenity patterns.
- Engineer meaningful features from the raw data.
- Segment PG listings using K-Means clustering.
- Select the number of clusters using the elbow method and silhouette score.
- Compare multiple classification models for predicting the generated segments.
- Deploy the trained pipeline through a Streamlit application.
- Explore potential business applications such as tenant comparison, market segmentation, and pricing guidance.

---

📊 Dataset

The project uses a Bengaluru PG listings dataset.

Dataset Size

- Raw dataset: 3,000 listings and 44 attributes
- After cleaning: 2,998 rows × 40 columns
- Used for modelling: 2,992 listings

Two duplicate rows were removed and four columns were dropped because they were not required for the planned analysis. Six additional rows containing an unresolved sentinel rent value ("999999") were excluded from modelling.

Major Feature Groups

Location & Identity

- PG Name
- Area
- Latitude / Longitude
- Pincode
- Nearest Metro
- Distance to Metro
- Nearest IT Park
- Distance to IT Park

Listing Attributes

- Gender
- Room Type
- Food Type
- Occupancy Status

Cost

- Monthly Rent
- Security Deposit
- Electricity Charges
- Water Charges
- Maintenance Fee

Quality Signals

- Google Rating
- Review Count
- Safety Score
- Commute Score
- Value for Money

Amenities

- WiFi
- Laundry
- Parking
- Power Backup
- Housekeeping
- Gym
- Lift
- CCTV
- RO Water
- AC
- TV
- Study Table
- Geyser
- and other amenity indicators

---

🧹 Data Cleaning

The raw dataset contained missing values, inconsistent categories, and corrupted numerical values.

The preprocessing included:

- Handling missing values.
- Standardizing inconsistent categorical values.
- Cleaning numeric fields such as rent and security deposit.
- Handling invalid Google Rating values outside the valid 0–5 range.
- Removing duplicate records.
- Converting amenity fields into consistent binary indicators.
- Handling unresolved sentinel rent values.
- Preparing the final modelling dataset.

---

⚙️ Feature Engineering

Two important features were engineered from the raw data.

1. Amenity_Count

Counts the number of available amenities across the amenity columns.

Amenity_Count =
Number of available amenities

2. Total_Monthly_Cost

Represents the estimated monthly cost beyond rent alone.

Total_Monthly_Cost =
Monthly_Rent
+ Electricity_Charges
+ Water_Charges
+ Maintenance_Fee

These engineered features provide more meaningful representations of overall amenities and monthly cost.

---

📐 Feature Scaling

Different numerical features have very different scales, so scaling was performed before clustering and classification.

RobustScaler

Used for cost-related features:

- Monthly_Rent
- Security_Deposit
- Total_Monthly_Cost

RobustScaler was used because these features are skewed and contain outliers.

StandardScaler

Used for:

- Google_Rating
- Review_Count
- Distance_to_Metro_km
- Distance_to_IT_Park_km
- Safety_Score
- Commute_Score
- Value_for_Money
- Amenity_Count

---

🤖 Machine Learning Approach

Stage 1 — K-Means Clustering

K-Means was used because the dataset did not contain pre-existing market segment labels.

The clustering model used 11 numerical features representing:

- Cost
- Safety
- Commute
- Location
- Quality
- Amenities

Each PG listing is represented as a point in the scaled feature space, and K-Means groups listings with similar feature profiles.

Selecting the Number of Clusters

Different values of K were evaluated using:

- Elbow method
- Silhouette score

The silhouette score was highest at:

K = 2 → 0.0801

However, K=2 primarily represented a simple low-cost versus high-cost separation.

K=4 produced more useful business-oriented segmentation while only slightly reducing the silhouette score:

K = 4 → 0.0718

Therefore, K=4 was selected as a business-driven trade-off between cluster separation and interpretability.

---

🏷️ Market Segments

After clustering, the average feature profile of each cluster was analyzed and interpreted.

Cluster 0 — Premium: High Cost, High Safety

Characteristics:

- Higher cost
- Higher safety
- Premium positioning

Listings: 513
Share: ~17.15%

Cluster 1 — Budget: Safe & Well Connected

Characteristics:

- Lower cost
- Relatively good safety
- Good connectivity

Listings: 868
Share: ~29.01%

Cluster 2 — Affordable: Great Commute, Low Safety

Characteristics:

- Affordable cost
- Strong commute characteristics
- Comparatively lower safety

Listings: 778
Share: ~26.00%

Cluster 3 — Affordable: Remote & Poorly Connected

Characteristics:

- Affordable cost
- Weaker connectivity
- More remote positioning

Listings: 833
Share: ~27.84%

«The cluster names are business interpretations of the feature profiles. They were not predefined labels given to the K-Means model.»

---

📊 Stage 2 — Classification

The cluster labels generated by K-Means were used as the target variable for supervised learning.

Three classification models were compared:

- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)

Model Performance

Model| Test Accuracy| Macro F1
Logistic Regression| 99.00%| 0.9885
Random Forest| 92.15%| 0.9194
KNN| 82.80%| 0.8234

Best Model

Logistic Regression was selected as the final classifier.

The high performance is partly explained by the fact that the target labels were generated by K-Means. The classifier learned to reproduce those cluster assignments effectively.

---

🔄 End-to-End Workflow

Raw PG Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Feature Scaling
       ↓
K-Means Clustering
       ↓
Cluster Profiling
       ↓
Generated Cluster Labels
       ↓
Train Classification Models
       ↓
Model Comparison
       ↓
Select Logistic Regression
       ↓
Save Trained Models
       ↓
Streamlit Application
       ↓
Predict Segment for New PG Listing

---

🖥️ Streamlit Application

The trained ML pipeline is integrated into a Streamlit application.

The application loads:

preprocessor.pkl
kmeans_model.pkl
cluster_classifier.pkl

A user can enter the characteristics of a new PG listing, and the application processes the input and predicts its market segment.

---

💼 Business Applications

The project can support several business use cases.

For PG Tenants

- Compare PGs beyond rent alone.
- Identify listings based on cost, safety, and commute characteristics.
- Reduce the time required to compare multiple listings.

For PG Operators

- Understand where their listing fits within the market.
- Compare listings with similar market segments.
- Support pricing and positioning decisions.
- Identify potential areas for improvement.

For PG Marketplaces

- Segment PG inventory.
- Build market intelligence dashboards.
- Support personalized search and recommendation systems.
- Analyze market-level differences between listings.

---

⚠️ Limitations

The clustering structure is relatively weak.

The final K-Means model achieved a silhouette score of approximately 0.0718, indicating weak natural separation between clusters.

Therefore, the four segments should be interpreted as business-oriented market segments, rather than strongly separated natural groups.

The 99% classification accuracy should also be interpreted carefully because the classification target was generated by K-Means rather than being an independently observed ground-truth label.

Before commercial deployment, the segmentation should be validated against real-world outcomes such as user behavior, bookings, occupancy, demand, and pricing.

---

🚀 Future Improvements

Potential improvements include:

- Collecting larger and more reliable real-world PG data.
- Adding historical pricing and demand information.
- Adding occupancy and booking information.
- Incorporating actual travel time instead of only distance.
- Improving safety-related data.
- Validating clusters against real user behavior.
- Building a personalized PG recommendation system.
- Adding model and data drift monitoring.
- Developing a business analytics dashboard.
- Performing A/B testing to measure whether segmentation improves user conversion.

---

🛠️ Tech Stack

Programming

- Python

Data Analysis

- Pandas
- NumPy

Visualization

- Matplotlib
- Seaborn
- Plotly

Machine Learning

- Scikit-learn
- K-Means
- Logistic Regression
- Random Forest
- KNN

Deployment

- Streamlit

Model Persistence

- Joblib

---

📁 Project Structure

bengaluru-pg-market-analysis/
│
├── app/
│   └── app1.py
│
├── data/
│   ├── Final_PG_Dataset.csv
│   └── Raw_PG_Dataset.csv
│
├── models/
│   ├── preprocessor.pkl
│   ├── kmeans_model.pkl
│   └── cluster_classifier.pkl
│
├── notebooks/
│   └── PG_Market_Analysis.ipynb
│
├── presentation/
│   └── Bengaluru_PG_Market_Analysis (2).pptx
│
├── README.md
├── requirements.txt
└── .gitignore

---

▶️ How to Run

1. Clone the repository

git clone https://github.com/BhumiPaliwal/bengaluru-pg-market-analysis.git

2. Navigate to the project directory

cd bengaluru-pg-market-analysis

3. Install dependencies

pip install -r requirements.txt

4. Run the Streamlit application

streamlit run app/app1.py

---

📌 Key Takeaway

This project demonstrates an end-to-end machine learning workflow:

Data Cleaning → EDA → Feature Engineering → Clustering → Classification → Evaluation → Model Persistence → Streamlit Deployment

The main objective is not simply to achieve a high model score, but to convert heterogeneous PG listing data into interpretable market segments that can support practical decision-making.

---

👩‍💻 Team
- BHUMI PALIWAL 
- MOHIT MALI
- GOURI SHRIMALI
- TANISHKA JAIN

B.Tech- Computer Science Engineering
