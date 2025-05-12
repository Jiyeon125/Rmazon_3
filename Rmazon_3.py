import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 📂 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_w_productName.csv')
    df = df.dropna(subset=['about_product', 'discounted_price', 'rating', 'rating_count', 'discount_percentage'])
    return df

df = load_data()

st.title("📊 유사 제품 추천 및 비교 도구")

# 📝 판매자 입력 받기
product_desc = st.text_area("제품 설명 입력", value="wireless bluetooth earbuds with noise cancelling and long battery life")
price = st.number_input("할인가", min_value=0, value=2499)
rating = st.slider("평점", 0.0, 5.0, 4.2)
review_count = st.number_input("리뷰 수", min_value=0, value=150)
discount_pct = st.slider("할인율 (%)", 0, 100, 20)

if st.button("유사 제품 추천"):
    # Step 1: TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['about_product'])
    query_vec = tfidf.transform([product_desc])
    cos_sim = cosine_similarity(query_vec, tfidf_matrix)

    top_indices = cos_sim[0].argsort()[::-1][:50]
    candidate_df = df.iloc[top_indices].copy()

    # Step 2: KMeans
    num_cols = ['discounted_price', 'rating', 'rating_count', 'discount_percentage']
    X = candidate_df[num_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=0)
    candidate_df['cluster'] = kmeans.fit_predict(X_scaled)

    input_features = [[price, rating, review_count, discount_pct]]
    input_scaled = scaler.transform(input_features)
    input_cluster = kmeans.predict(input_scaled)[0]

    cluster_members = candidate_df[candidate_df['cluster'] == input_cluster]
    member_scaled = scaler.transform(cluster_members[num_cols])

    dists = euclidean_distances(input_scaled, member_scaled)[0]
    cluster_members = cluster_members.copy()
    cluster_members['distance'] = dists

    top_matches = cluster_members.sort_values('distance').head(3)

    st.subheader("📋 유사한 상위 3개 제품")
    st.dataframe(top_matches[['product_name', 'discounted_price', 'rating', 'rating_count', 'discount_percentage', 'distance']])
