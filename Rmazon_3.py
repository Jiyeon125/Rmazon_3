# 📦 예비 상품 판매자용 Streamlit 앱
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 📂 데이터 로딩 및 정가 계산
def load_data():
    df = pd.read_csv("cleaned_amazon_0519.csv")
    df = df.dropna(subset=['about_product', 'discounted_price', 'discount_percentage'])
    df['actual_price'] = df['discounted_price'] / (1 - df['discount_percentage'] / 100)
    df[['cat1', 'cat2', 'cat3']] = df['category'].str.split('|', expand=True, n=2)
    return df

@st.cache_data
def load_tokenizer_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    return tokenizer, model

def t5_summarize(text, max_length=50):
    input_text = "summarize: " + text.strip().replace("\n", " ")
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ 데이터 준비
st.title("🧭 예비 판매자를 위한 시장 유사 제품 탐색기")
st.markdown("*앱 첫 구동 시 라이브러리 다운로드로 인해 로딩이 오래 걸릴 수 있습니다.")
df = load_data()
tokenizer, model = load_tokenizer_model()

# 🔍 카테고리 드릴다운 선택
cat1 = st.selectbox("1차 카테고리", sorted(df['cat1'].dropna().unique()))
cat2_options = df[df['cat1'] == cat1]['cat2'].dropna().unique()
cat2 = st.selectbox("2차 카테고리", sorted(cat2_options))
cat3_options = df[(df['cat1'] == cat1) & (df['cat2'] == cat2)]['cat3'].dropna().unique()
cat3 = st.selectbox("3차 카테고리", ["전체"] + sorted(cat3_options))

# ✅ 최종 카테고리 필터링
if cat3 == "전체":
    df_filtered = df[(df['cat1'] == cat1) & (df['cat2'] == cat2)]
else:
    df_filtered = df[(df['cat1'] == cat1) & (df['cat2'] == cat2) & (df['cat3'] == cat3)]

# 📝 사용자 입력
product_desc = st.text_area("제품 설명 입력", placeholder="예시: Outdoor camping gear with solar panel")
actual_price = st.number_input("정가 (₹)", min_value=0, value=3000)
discount_pct = st.slider("할인율 (%)", 0, 100, 20)
discounted_price = int(actual_price * (1 - discount_pct / 100))
st.markdown(f"**할인가 (자동 계산): ₹{discounted_price}**")

# ▶️ 실행 버튼
if st.button("유사 제품 탐색하기"):
    if len(df_filtered) < 5:
        st.error("선택한 카테고리 내 제품 수가 너무 적습니다.")
    else:
        # 🔎 TF-IDF + 코사인 유사도
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df_filtered['about_product'])
        query_vec = tfidf.transform([product_desc])
        cos_sim = cosine_similarity(query_vec, tfidf_matrix)

        top_indices = cos_sim[0].argsort()[::-1][:50]
        candidate_df = df_filtered.iloc[top_indices].copy()

        # 🎯 유사도 진단
        mean_sim = cos_sim[0][top_indices].mean()
        max_sim = cos_sim[0][top_indices].max()
        sim_warnings = []
        if mean_sim < 0.05:
            sim_warnings.append("⚠️ 입력한 설명이 다른 제품들과 전반적으로 크게 다릅니다. (평균 유사도 낮음)")
        if max_sim < 0.1:
            sim_warnings.append("⚠️ 입력한 설명과 매우 유사한 제품이 거의 없습니다. (최고 유사도 낮음)")

        # 📊 KMeans 클러스터링 (k=4)
        num_cols = ['actual_price', 'discount_percentage']
        X = candidate_df[num_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=4, random_state=0)
        candidate_df['cluster'] = kmeans.fit_predict(X_scaled)

        input_scaled = scaler.transform([[actual_price, discount_pct]])
        input_cluster = kmeans.predict(input_scaled)[0]

        cluster_members = candidate_df[candidate_df['cluster'] == input_cluster]
        member_scaled = scaler.transform(cluster_members[num_cols])
        dists = euclidean_distances(input_scaled, member_scaled)[0]
        cluster_members = cluster_members.copy()
        cluster_members['distance'] = dists
        top_matches = cluster_members.sort_values('distance').head(3).reset_index(drop=True)

        # ⚠️ 유사도 경고 출력
        if sim_warnings:
            st.warning("\n\n".join(sim_warnings))

        # 📋 결과 출력
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
                # 개별 리뷰 요약
                summary_text = row.get("full_summary", "")
                if pd.notna(summary_text) and summary_text.strip():
                    with st.spinner("AI가 해당 제품 리뷰 요약 중..."):
                        summary = t5_summarize(summary_text, max_length=50)
                        st.markdown(f"🧠 **AI 리뷰 요약:** {summary}")
                else:
                    st.markdown("🧠 **AI 리뷰 요약:** (리뷰 요약 없음)")

        # 전체 리뷰 요약
        all_text = " ".join(top_matches['full_summary'].dropna().astype(str).tolist())
        if all_text.strip():
            with st.spinner("AI가 상위 제품들의 리뷰 전체 요약 중..."):
                full_summary = t5_summarize(all_text, max_length=100)
                st.subheader("🧠 상위 제품 리뷰 전체 요약")
                st.markdown(f"> {full_summary}")
