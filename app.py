import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

import platform
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Linux':
    rc('font', family='NanumGothic')

def main() :
    st.title('K-Means 클러스터링 앱')
    st.text('csv 파일을 업로드 하면, 비슷한 유형의 데이터끼리 묶어주는 앱입니다.')
    
    # 1. csv 파일 업로드
    file = st.file_uploader('CSV 파일 업로드', type=['csv'])

    if file is not None :

        # 1-1. 판다스의 데이터프레임으로 읽는다.
        df = pd.read_csv(file)

        # 1-2. 10개 미만의 파일을 올리면, 에러 처리 하자.
        print(df.shape)
        if df.shape[0] < 10 :
            st.error('데이터의 갯수는 10개 이상이어야 합니다.')
            return

        # 1-3. 유저한테 데이터프레임 보여준다.
        st.dataframe(df)

        # 2. nan 데이터 있으면, 삭제하자.
        # print(df.isna().sum())

        st.subheader('각 컬럼별 NaN의 갯수입니다.')

        st.dataframe(df.isna().sum())

        df.dropna(inplace=True)

        df.reset_index(inplace=True, drop=True)

        st.info('NaN이 있으면 해당 데이터는 자동으로 삭제합니다.')
    
        # 3. 유저한테 컬럼을 선택할수 있도록 하자.
        print(df.columns)

        st.subheader('클러스터링에 사용할 컬럼 선택!')

        seleted_columns = st.multiselect('X로 사용할 컬럼을 선택하세요', df.columns)

        X = df[seleted_columns]

        if len(seleted_columns) >= 2 :

            st.dataframe(X)
            
            X_new = pd.DataFrame()

            print(X_new)
            
            # 4. 해당 컬럼의 데이터가 문자열이면, 숫자로 바꿔주자.

            for column in X.columns :

                print( X[column].dtype )

                # 컬럼의 데이터가 문자열이면, 레이블인코딩 또는 원핫인코딩해야 한다.
                if X[column].dtype == object :
                    if X[column].nunique() >= 3 :
                        #원핫 인코딩
                        column_names = sorted( X[column].unique() )
                        # 비어있는 데이터프레임에 컬럼 추가
                        X_new[column_names] = pd.get_dummies( X[column].to_frame() )
                        
                    else :
                        #레이블 인코딩
                        
                        encoder = LabelEncoder()
                        X_new[column] = encoder.fit_transform( X[column] )
                        
                else :
                    # 숫자 데이터 처리
                    X_new[column] = X[column]

            # X_new 는 숫자로만 되어있는 데이터프레임.
            # 4-1. 유저한테 보여주자.
            
            X_new.reset_index(inplace=True, drop=True)
            
            st.subheader('클러스터링에 실제 사용할 데이터')
            st.dataframe( X_new )

            # # 5. k의 갯수를 1개부터 10개까지 해서 wcss 를 구한다.
            # 해당 코드는 데이터를 처음 넣었을때 처리하는게 더 유용하기 때문에 1-2로 바꾸어서 올리자.
            # 이런식으로 로직을 풀어나가다가 수정사항이 있으면 유기적으로 바꾸면 된다.
            # if X.shape[0] < 10 : => 상단으로 올리기때문에 위에서는 X를 df로 바꾼다. 상단에선 X가 없기때문에..
            #     st.error('데이터의 갯수는 10개 이상이어야 합니다.')
            wcss = []
            for k in np.arange(1, 10 + 1) :
                kmeans = KMeans(n_clusters = k, random_state=5)
                kmeans.fit(X_new)
                wcss.append(kmeans.inertia_)


            # 6. elbow method 를 이용해서, 차트로 보여준다.
            fig1 = plt.figure()
            X = np.arange(1, 10+1)
            plt.plot(X, wcss)
            plt.title('엘보우 메소드')
            plt.xlabel('클러스터의 갯수')
            plt.ylabel('WCSS')
            st.pyplot(fig1)

            # 7. 유저가 k의 갯수를 정한다.
            k = st.slider('클러스터 갯수 설정', 1, 10)

            # 8. KMeans 수행해서 그룹정보를 가져온다.
            kmeans = KMeans(n_clusters= k, random_state= 5)
            y_pred = kmeans.fit_predict(X_new)

            # 9. 원래 있던 df 에 Group 이라는 컬럼을 만들어준다.
            df['Group'] = y_pred

            # 10. 결과를 파일로 저장한다.
            df.to_csv('result_csv')

            # 11. 유저에게 보여준다.
            st.subheader('클러스터링 결과')

            st.dataframe(df)

            # 12. 유저가 그룹을 선택하면, 해당 그룹의 정보를 보여준다.
            choice = st.selectbox('그룹을 선택하세요', np.arange(0, k))
            
            st.dataframe(df.loc[ df['Group'] == choice , ])


if __name__ == '__main__' :
    main()