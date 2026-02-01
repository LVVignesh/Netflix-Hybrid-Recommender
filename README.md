# 🎬 Netflix Hybrid Recommendation Engine

### **Overview**
This project implements a **Collaborative Filtering** recommendation system utilizing **Matrix Factorization (SVD)**. It is designed to handle the "cold start" problem by simulating user interactions through implicit feedback—specifically combining item popularity, release year, and runtime to generate behavioral signals.

This repository serves as a professional demonstration of information retrieval, dimensionality reduction, and rank-based evaluation metrics within a production-style machine learning pipeline.



---

### **🚀 Core Architecture**

#### **1. Implicit Feedback Simulation**
* Generates a synthetic user-item interaction matrix for 100 users.
* Engineers a "popularity" score by ranking items based on release year and runtime, serving as a proxy for user preference in the absence of explicit rating data.

#### **2. Dimensionality Reduction (Truncated SVD)**
* Reduces a sparse user-item matrix of 3,599 titles into **30 Latent Factors**.
* Captures approximately **36.1% of the total variance**, identifying hidden relationships between content types (e.g., genre patterns, thematic similarities).

#### **3. Similarity Engine**
* Computes a **Cosine Similarity** matrix across movie latent factors.
* Enables real-time retrieval of the top-K similar items based on behavioral patterns rather than just metadata.

---

### **📊 Performance Evaluation**
Recommendation systems require rank-aware metrics rather than standard accuracy. This project evaluates performance using **Precision@K** and **Recall@K** to measure the relevance of the top-10 suggested items.

* **Precision@10: 0.397** — Approximately 40% of the top-10 recommendations matched items in the simulated users' watch history.
* **Recall@10: 0.033** — Measures the proportion of total relevant items successfully captured within a constrained top-10 list.
* **Variance Explained: 0.361** — Indicates the efficiency of the Truncated SVD in preserving information while reducing dimensionality.



---

### **🛠️ Technical Stack**
* **Language:** Python 3.12+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
    * `TruncatedSVD`: Matrix Factorization
    * `MinMaxScaler`: Feature Scaling
    * `cosine_similarity`: Similarity Mapping

---

### **📂 Project Structure**
* `netflix_recommendation_system.py`: Master script containing the pipeline from data cleaning to evaluation.
* `netflix.csv`: Raw dataset containing movie metadata (title, type, runtime, etc.).
* `.gitignore`: Python-specific exclusion rules.
* `README.md`: Project documentation.

---

### **🚀 How to Run**
1. Ensure `netflix.csv` is in the root directory.
2. Install dependencies: 
   ```bash
   pip install pandas numpy scikit-learn