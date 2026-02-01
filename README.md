# 🎬 Netflix Hybrid Recommendation Engine (ML-Based)

## 📌 Project Overview
This project implements a **Hybrid Recommendation System** for Netflix-style content using classical Machine Learning techniques. It demonstrates the transition from raw metadata to behavioral intelligence by combining **Collaborative Filtering**, **Matrix Factorization (SVD)**, and **Latent Feature Discovery**.

The core objective of this project is to simulate how real-world production systems (like Netflix or Spotify) identify hidden relationships between users and items using a "Candidate Generation" and "Ranking" logic, without relying on heavy deep learning frameworks.

---

## 🧠 The Hybrid Logic: Why This Approach?
Unlike simple content-based filtering, this system is a **Hybrid** because it bridges the gap between metadata and behavior:
* **Discovery (Unsupervised):** Uses **Truncated SVD** to break down a sparse user-item matrix into latent factors, discovering hidden themes (e.g., "Gritty Korean Crime Thrillers") that aren't explicitly labeled.
* **Ranking (Performance):** Engineers a **Popularity/Quality Signal** based on release year and runtime to rank the most relevant "hits" for a specific user.

This architecture effectively addresses the **Cold Start problem** by using content signals to generate recommendations even when explicit user rating data is limited.

---

## 🏗️ Project Architecture

```text
Data Loading (netflix.csv)
   ↓
Feature Engineering (Runtime normalization & Popularity Scaling)
   ↓
Interaction Simulation (Synthetic User-Item Matrix)
   ↓
Matrix Factorization (Truncated SVD)
   ↓
Latent Feature Learning (30 Components)
   ↓
Cosine Similarity (Item-to-Item Mapping)
   ↓
Evaluation (Precision@K, Recall@K)

🚀 Key Features

User–Item Interaction Matrix: Simulated implicit feedback loop for 100 users.
Latent Factor Discovery: Decomposition of 3,599 titles into 30 dense feature vectors.
Variance Capture: Successfully captures ~36% of data structure through dimensionality reduction.
Rank-Based Metrics: Implementation of industry-standard Precision and Recall at K ($K=10$).

Metric,Score,Description
Precision@10,0.397,~40% of the top-10 recommendations were relevant to the user.
Recall@10,0.033,The proportion of total relevant items captured in the top-10.
Variance Explained,0.361,The amount of interaction structure captured by SVD components.


🛠️ Technical Stack

Language: Python 3.12+

Data Analysis: Pandas, NumPy

Machine Learning: Scikit-Learn

TruncatedSVD (Matrix Factorization)

MinMaxScaler (Feature Scaling)

cosine_similarity (Vector Space Mapping)



🧪 Sample Recommendation Output

Input: #Alive (Korean Thriller)

Output:

Hey Arnold! The Jungle Movie

Night in Paradise (Korean Crime/Thriller)

Dovlatov

Todd Glass: Act Happy

Long Story Short

Note: The system correctly identified 'Night in Paradise' as a behavioral match to '#Alive' through latent factor similarity.

📂 Project Structure

Netflix Recommendation system.py: The full ML pipeline script.

netflix.csv: Raw metadata for movies and shows.

.gitignore: Standard Python exclusions.

README.md: Project documentation.

🧑‍💻 How to Run

Clone the repository:git clone [https://github.com/LVVignesh/Netflix-Hybrid-Recommender.git](https://github.com/LVVignesh/Netflix-Hybrid-Recommender.git)

Install dependencies:

Bash
pip install pandas numpy scikit-learn
Run the engine:

Bash
python "Netflix Recommendation system.py"

FUTURE ENHANCEMENTS

Neural Collaborative Filtering: Replacing SVD with Embedding layers.

Hybrid Deep Learning: Using Autoencoders to reconstruct missing ratings.

Explainable AI (XAI): Integrating SHAP to explain why a specific movie was recommended.


Author: Vignesh LV