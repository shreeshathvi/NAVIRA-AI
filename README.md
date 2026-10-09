# 🚀 NAVIRA AI — Intelligent Journey. Safer Decisions.



🌐 **Live Demo:** https://navira-ai-pink.vercel.app/


## 📌 Project Overview

NAVIRA AI is an AI-powered travel risk and route intelligence web application designed to help users explore alternative travel routes and understand potential road accident risks.

The project combines route planning, map visualization, machine learning, and journey recommendations in one web application.

It uses historical road accident data to explore accident severity and patterns. Route information is retrieved through a routing API, while machine learning models support accident analysis.



 Key Features

* 🗺️ **Route Planning:** Explore available routes between locations.
* 📍 **Map Visualization:** View routes on an interactive map.
* 📊 **Travel Risk Intelligence:** Present accident-related risk insights.
* 🤖 **Accident Severity Classification:** Apply Random Forest classification.
* 🔍 **Accident Pattern Clustering:** Use K-Means to group accident records.
* 💡 **Journey Recommendations:** Display route-related recommendations.
* 💬 **Chatbot Interface:** Provide an interactive way to ask travel-related questions.



1. Random Forest — Supervised Learning

Random Forest combines multiple decision trees to perform classification. In NAVIRA AI, it is used to classify road accident severity based on selected accident-related features.

**Purpose:**

* Learn from labeled accident records.
* Predict accident severity classes.
* Evaluate classification performance using metrics such as accuracy and F1-score.


K-Means groups accident records into clusters based on similarities in the selected features.

**Purpose:**

* Explore patterns in road accident data.
* Group similar accident records.
* Visualize clusters to support exploratory analysis.

These clusters represent patterns in the dataset; they do not automatically establish real-world accident hotspots.

## 🛠️ Technologies Used

| Technology      | Purpose                                 |
| --------------- | --------------------------------------- |
| Python          | Main programming language               |
| Flask           | Web application backend                 |
| Pandas          | Data processing and analysis            |
| NumPy           | Numerical operations                    |
| Scikit-learn    | Machine learning algorithms             |
| Joblib          | Saving and loading trained models       |
| HTML & CSS      | Web page structure and styling          |
| JavaScript      | Frontend interactions                   | |

