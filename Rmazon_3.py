import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 📂 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_amazon_0519.csv')
    df = df.dropna(subset=['about_product', 'discounted_price', 'discount_percentage'])
    return df

df = load_data()

st.title("🧭 예비 판매자를 위한 시장 내 유사 상품 탐색기")

# 🔍 카테고리 자동완성
category_list = sorted(df['category'].dropna().unique().tolist())
typed = st.text_input("카테고리 검색", "")
filtered_categories = [cat for cat in category_list if typed.lower() in cat.lower()]
selected_category = st.selectbox("카테고리 선택", filtered_categories) if filtered_categories else None

# 💬 제품 설명 입력 + 가격/할인율
product_desc = st.text_area("상품 설명 입력", placeholder="예시: Outdoor camping gear with solar panel")
actual_price = st.number_input("정가 (₹)", min_value=0, value=3000)
discount_pct = st.slider("할인율 (%)", 0, 100, 20)

# 💸 자동 계산된 할인가 표시
discounted_price = int(actual_price * (1 - discount_pct / 100))
st.markdown(f"**할인가 (자동 계산): ₹{discounted_price}**")

# ▶️ 버튼 클릭 시 실행
if st.button("시장 내 유사 상품 탐색하기"):
    if selected_category is None:
        st.warning("카테고리를 먼저 검색 후 선택해 주세요.")
    else:
        df_filtered = df[df['category'] == selected_category]

        if len(df_filtered) < 5:
            st.error("선택한 카테고리 내 제품 수가 너무 적습니다. 다른 카테고리를 선택해 주세요.")
        else:
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

                # ✅ 클러스터링
                num_cols = ['actual_price', 'discount_percentage']
                candidate_df['actual_price'] = candidate_df['discounted_price'] / (1 - candidate_df['discount_percentage'] / 100)
                X = candidate_df[['actual_price', 'discount_percentage']]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                k = min(5, len(candidate_df))
                kmeans = KMeans(n_clusters=k, random_state=0)
                candidate_df['cluster'] = kmeans.fit_predict(X_scaled)

                input_features = [[actual_price, discount_pct]]
                input_scaled = scaler.transform(input_features)
                input_cluster = kmeans.predict(input_scaled)[0]

                cluster_members = candidate_df[candidate_df['cluster'] == input_cluster]
                member_scaled = scaler.transform(cluster_members[['actual_price', 'discount_percentage']])
                dists = euclidean_distances(input_scaled, member_scaled)[0]
                cluster_members = cluster_members.copy()
                cluster_members['distance'] = dists

                top_matches = cluster_members.sort_values('distance').head(3).reset_index(drop=True)

                # ⚠️ 경고 먼저 출력
                if similarity_warnings:
                    st.warning("\n\n".join(similarity_warnings))

                # ✅ 카드 형태 결과 출력
                st.subheader("📋 유사한 상위 3개 제품")

                for i, row in top_matches.iterrows():
                    st.markdown(f"### {i+1}위. {row['product_name']}")
                    cols = st.columns([1, 3])
                    with cols[0]:
                        st.image(row['img_link'], width=120)
                    with cols[1]:
                        st.markdown(f"**Distance**: `{row['distance']:.4f}`")
                        st.markdown(f"`정가`: ₹{int(row['actual_price'])} / `할인율`: {int(row['discount_percentage'])}% / `할인가`: ₹{int(row['discounted_price'])}")
                        st.markdown(f"`평점`: {row.get('rating', 'N/A')} ⭐ / `리뷰 수`: {row.get('rating_count', 'N/A')}")
