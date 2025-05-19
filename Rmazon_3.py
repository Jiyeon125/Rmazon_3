import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 📂 데이터 불러오기 + 결측치 제거
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_w_productName.csv')
    df = df.dropna(subset=['about_product', 'discounted_price', 'rating', 'rating_count', 'discount_percentage'])
    return df

df = load_data()

# 🧱 앱 제목
st.title("📊 시장 내 유사 제품 탐색기")

# 🔍 카테고리 자동완성 검색 기능
category_list = sorted(df['category'].dropna().unique().tolist())  # 고유 카테고리 목록
typed = st.text_input("카테고리 검색", "")  # 검색어 입력

# 검색어가 포함된 카테고리만 필터링
filtered_categories = [cat for cat in category_list if typed.lower() in cat.lower()]
selected_category = st.selectbox("카테고리 선택 (자동완성)", filtered_categories) if filtered_categories else None

# 📝 판매자 제품 정보 입력 UI
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
        # ✅ Step 1: 선택된 카테고리 기준 필터링
        df_filtered = df[df['category'] == selected_category]

        # 해당 카테고리 내 제품 수 확인
        if len(df_filtered) < 5:
            st.error("선택한 카테고리 내 제품 수가 너무 적습니다. 다른 카테고리를 선택해 주세요.")
        else:
            # ✅ Step 2: 제품 설명 기반 TF-IDF 유사도 계산
            tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = tfidf.fit_transform(df_filtered['about_product'])

            # 입력 설명 벡터화 및 유사도 계산
            query_vec = tfidf.transform([product_desc])
            cos_sim = cosine_similarity(query_vec, tfidf_matrix)

            # 상위 유사도 50개 제품 추출 (단, 실제 개수보다 많으면 제한됨)
            top_n = min(50, len(df_filtered))
            top_indices = cos_sim[0].argsort()[::-1][:top_n]
            candidate_df = df_filtered.iloc[top_indices].copy()

            # 예외처리: 후보군이 너무 적을 경우 종료
            if len(candidate_df) < 3:
                st.error("유사한 제품이 3개 미만입니다. 다른 설명을 입력하거나 다른 카테고리를 선택해 주세요.")
            else:
                # ✅ Step 3: 수치형 특성 기반 KMeans 클러스터링
                num_cols = ['discounted_price', 'rating', 'rating_count', 'discount_percentage']
                X = candidate_df[num_cols]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # 클러스터 수는 데이터 수보다 많지 않게 조정
                k = min(5, len(candidate_df))
                kmeans = KMeans(n_clusters=k, random_state=0)
                candidate_df['cluster'] = kmeans.fit_predict(X_scaled)

                # 입력 제품을 동일한 방식으로 스케일링
                input_features = [[price, rating, review_count, discount_pct]]
                input_scaled = scaler.transform(input_features)
                input_cluster = kmeans.predict(input_scaled)[0]

                # ✅ Step 4: 동일 클러스터 내 유사 제품 추출
                cluster_members = candidate_df[candidate_df['cluster'] == input_cluster]
                member_scaled = scaler.transform(cluster_members[num_cols])

                # 거리 계산 후 정렬
                dists = euclidean_distances(input_scaled, member_scaled)[0]
                cluster_members = cluster_members.copy()
                cluster_members['distance'] = dists

                top_matches = cluster_members.sort_values('distance').head(3)

                # ✅ 결과 출력
                st.subheader("📋 가장 유사한 상위 3개 제품")
                st.dataframe(top_matches[['product_name', 'discounted_price', 'rating', 'rating_count', 'discount_percentage', 'distance']])
                st.caption("📌 Distance 값이 작을수록 입력 제품과 수치적으로 유사한 제품입니다.")
