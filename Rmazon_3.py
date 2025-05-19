import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 📂 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_amazon_0519.csv')  # ✅ 파일명 반영
    df = df.dropna(subset=['about_product', 'discounted_price', 'rating', 'rating_count', 'discount_percentage'])
    return df

df = load_data()

# 🧱 앱 제목
st.title("📊 시장 내 유사 제품 탐색기")

# 🔍 카테고리 검색 + 자동완성 선택
category_list = sorted(df['category'].dropna().unique().tolist())
typed = st.text_input("카테고리 검색", "")
filtered_categories = [cat for cat in category_list if typed.lower() in cat.lower()]
selected_category = st.selectbox("카테고리 선택 (자동완성)", filtered_categories) if filtered_categories else None

# 📝 입력값
product_desc = st.text_area("제품 설명 입력", value="wireless bluetooth earbuds with noise cancelling and long battery life")
price = st.number_input("할인가", min_value=0, value=2499)
rating = st.slider("평점", 0.0, 5.0, 4.2)
review_count = st.number_input("리뷰 수", min_value=0, value=150)
discount_pct = st.slider("할인율 (%)", 0, 100, 20)

# ▶️ 버튼 클릭 시 실행
if st.button("유사 제품 탐색하기"):
    if selected_category is None:
        st.warning("카테고리를 먼저 검색 후 선택해 주세요.")
    else:
        df_filtered = df[df['category'] == selected_category]

        if len(df_filtered) < 5:
            st.error("선택한 카테고리 내 제품 수가 너무 적습니다. 다른 카테고리를 선택해 주세요.")
        else:
            # TF-IDF 기반 유사도 계산
            tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = tfidf.fit_transform(df_filtered['about_product'])
            query_vec = tfidf.transform([product_desc])
            cos_sim = cosine_similarity(query_vec, tfidf_matrix)

            top_n = min(50, len(df_filtered))
            top_indices = cos_sim[0].argsort()[::-1][:top_n]
            candidate_df = df_filtered.iloc[top_indices].copy()

            if len(candidate_df) < 3:
                st.error("유사한 제품이 3개 미만입니다. 설명을 다시 입력하거나 다른 카테고리를 선택해 주세요.")
            else:
                # 🎯 유사도 진단
                mean_sim = cos_sim[0][top_indices].mean()
                max_sim = cos_sim[0][top_indices].max()

                similarity_warnings = []
                if mean_sim < 0.05:
                    similarity_warnings.append("⚠️ 입력한 설명이 다른 제품들과 전반적으로 크게 다릅니다. 유사 제품 목록의 정확도가 낮을 수 있습니다. (평균 유사도 낮음)\n권장: 설명을 더 구체적으로 작성해 보세요.")
                if max_sim < 0.1:
                    similarity_warnings.append("⚠️ 입력한 설명과 매우 유사한 제품이 거의 없습니다. 유사 제품 목록의 정확도가 낮을 수 있습니다. (최고 유사도 낮음)")

                # 클러스터링 준비
                num_cols = ['discounted_price', 'rating', 'rating_count', 'discount_percentage']
                X = candidate_df[num_cols]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                k = min(5, len(candidate_df))
                kmeans = KMeans(n_clusters=k, random_state=0)
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

                # ✅ 결과 출력
                st.subheader("📋 가장 유사한 상위 3개 제품")
                st.dataframe(top_matches[['product_name', 'discounted_price', 'rating', 'rating_count', 'discount_percentage', 'distance']])
                st.caption("📌 Distance 값이 작을수록 입력 제품과 수치적으로 유사한 제품입니다.")

                # ⚠️ 유사도 진단 결과 메시지 출력
                if similarity_warnings:
                    st.warning("\n\n".join(similarity_warnings))
