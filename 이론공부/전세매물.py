
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd



# 매물 데이터 벡터화
# 영화 줄거리 대신, 각 매물의 속성 데이터를 벡터화할 수 있습니다. 각 속성을 숫자 또는 범주형 값으로 표현한 후, 이를 기반으로 벡터를 생성합니다.

# 예시:

# 지역: 지역을 숫자로 매핑 (예: 강남=1, 송파=2, 종로=3 등)
# 면적: 면적을 정규화된 숫자로 변환
# 가격: 가격을 일정한 구간으로 나누어 정규화
# 방 개수: 정수형 데이터
# 편의시설: 특정 키워드를 기반으로 0 또는 1로 표현 (예: 역세권=1, 비역세권=0)
# 이렇게 하면 각 전세 매물을 특성 벡터로 변환할 수 있습니다.

# 변수별 벡터화 방법 선택: 
# 각 변수(예: 가격, 위치, 면적, 방 개수 등)에 대해 
# 적절한 벡터화 방법을 사용해야 합니다.

# 범주형 데이터 (예: 지역, 건물 형태 등): 
# One-Hot Encoding 또는 Label Encoding을 사용할 수 있습니다.
# 수치형 데이터 (예: 가격, 면적, 방 개수 등): 
# 정규화 또는 표준화를 통해 값의 범위를 맞추는 것이 일반적입니다.
# 텍스트 데이터 (예: 설명, 특이 사항 등): 
# TF-IDF와 같은 기법을 사용하여 벡터화할 수 있습니다.


# 각 변수 벡터화: 각 변수를 벡터화한 후, 이 벡터들을 하나의 큰 벡터로 통합합니다.
# 예를 들어, 매물의 위치, 가격, 면적, 설명 등을 모두 벡터화한 후 이를 결합합니다.


# 데이터 예시 (지역, 가격, 면적, 설명)
data = {'지역': ['강남', '송파', '강북'],
        '가격': [300000, 250000, 200000],
        '면적': [85, 75, 65],
        '설명': ['넓고 쾌적한 아파트', '교통이 편리한 지역', '저렴한 아파트']}

df = pd.DataFrame(data)

# 1. 범주형 변수 (지역) 벡터화 (One-Hot Encoding)
one_hot = OneHotEncoder()
location_encoded = one_hot.fit_transform(df[['지역']]).toarray()

# 2. 수치형 변수 (가격, 면적) 벡터화 (정규화)
scaler = StandardScaler()
numeric_features = scaler.fit_transform(df[['가격', '면적']])

# 3. 텍스트 변수 (설명) 벡터화 (TF-IDF)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['설명']).toarray()

# 4. 모든 변수를 결합하여 하나의 벡터로
import numpy as np
final_matrix = np.hstack((location_encoded, numeric_features, tfidf_matrix))

print(final_matrix)



from sklearn.metrics.pairwise import cosine_similarity

# 매물 간 유사도 계산
similarity_matrix = cosine_similarity(final_matrix)

# 유사도 행렬 출력
print(similarity_matrix)




# 0번째 매물과 가장 유사한 매물 3개 추천 (자기 자신 제외)
similarity_scores = list(enumerate(similarity_matrix[0]))
# 지금은 0번째 매물이라고 했지만 
# 매물 아이디값? 을 직접 넣어주도록 함수의매개변수로 지정 


# 유사도 점수를 기준으로 내림차순 정렬
sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# 상위 3개의 유사한 매물 출력 (자기 자신 제외)
top_similar_properties = sorted_scores[1:4]
print("\n0번째 매물과 유사한 상위 3개 매물:\n", top_similar_properties)