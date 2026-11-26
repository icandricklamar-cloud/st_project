import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="NBA 3점슛과 승률 회귀 분석",
    page_icon="🏀",
    layout="wide",
)

# 2025-26 NBA 정규시즌 30개 팀 검증 데이터 (출처: Basketball-Reference)
RAW_CSV = """Team,3P,3PA,FG%,W/L%,PS/G,PA/G
덴버 너겟츠,14.2,35.8,0.496,0.659,122.1,116.9
마이애미 히트,13.7,37.9,0.468,0.524,120.9,118.5
샌안토니오 스퍼스,13.6,37.9,0.483,0.756,119.8,111.5
클리블랜드 캐벌리어스,14.3,39.8,0.482,0.634,119.5,115.4
오클라호마 시티 썬더,13.8,37.9,0.484,0.780,119.0,107.9
애틀랜타 호크스,14.6,39.5,0.474,0.561,118.5,116.0
미네소타 팀버울브스,13.8,37.2,0.481,0.598,118.0,114.6
디트로이트 피스턴스,11.0,30.9,0.485,0.732,117.8,109.6
유타 재즈,12.6,36.7,0.467,0.268,117.6,126.0
뉴욕 닉스,14.2,38.2,0.478,0.646,116.5,110.1
시카고 불스,14.3,40.1,0.469,0.378,116.3,121.5
로스앤젤레스 레이커스,11.8,32.9,0.502,0.646,116.3,114.6
샬럿 호니츠,16.4,43.3,0.460,0.537,116.0,111.2
필라델피아 76ers,12.3,35.3,0.462,0.549,115.9,116.1
올랜도 매직,11.7,34.1,0.464,0.549,115.7,115.1
뉴올리언스 펠리컨스,11.1,32.1,0.466,0.317,115.5,120.0
포틀랜드 트레일 블레이저스,14.5,42.2,0.453,0.512,115.5,115.8
휴스턴 로케츠,11.5,31.5,0.479,0.634,115.2,110.0
보스턴 셀틱스,15.5,42.1,0.467,0.683,114.9,107.2
멤피스 그리즐리스,13.9,39.5,0.456,0.305,114.7,120.7
골든 스테이트 워리어스,15.7,44.1,0.461,0.451,114.6,115.2
토론토 랩터스,11.4,32.1,0.482,0.561,114.6,111.8
댈러스 매버릭스,11.0,31.9,0.466,0.317,114.1,119.6
로스앤젤레스 클리퍼스,12.4,33.8,0.484,0.512,113.8,112.6
워싱턴 위저즈,13.0,36.4,0.463,0.207,112.9,124.9
피닉스 선즈,14.8,40.8,0.455,0.549,112.6,111.1
인디애나 페이서스,13.4,37.6,0.458,0.232,112.4,120.4
새크라멘토 킹스,10.3,30.2,0.467,0.268,111.0,121.0
밀워키 벅스,15.1,39.1,0.477,0.390,110.6,116.8
브루클린 네츠,13.1,38.4,0.443,0.244,105.9,115.9
"""

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(RAW_CSV))
    df["득실점차(PS-PA)"] = (df["PS/G"] - df["PA/G"]).round(1)
    return df

df = load_data()

# 사이드바 설정
st.sidebar.header("분석 설정")
selected_x = st.sidebar.selectbox(
    "독립변수(X) 선택",
    options=["3P", "3PA", "FG%", "PS/G", "PA/G", "득실점차(PS-PA)"],
    index=0,
    help="기본 분석 변수는 팀별 경기당 평균 3점슛 성공 개수(3P)입니다.",
)
highlight_teams = st.sidebar.multiselect(
    "그래프에 이름 표시할 팀 선택",
    options=df["Team"].tolist(),
    default=["샬럿 호니츠", "디트로이트 피스턴스", "오클라호마 시티 썬더", "워싱턴 위저즈"],
)
show_labels = st.sidebar.checkbox("모든 팀 이름 표시", value=False)

st.sidebar.download_button(
    label="📥 전처리 완료 CSV 다운로드 (30행)",
    data=RAW_CSV.encode("utf-8-sig"),
    file_name="nba_2025_26_3p_winrate.csv",
    mime="text/csv",
)

# 회귀 모델 계산 (선택된 X -> W/L%)
X = df[[selected_x]].values
y = df["W/L%"].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 베이스라인 (전체 팀 평균 승률로만 예측할 때)
baseline_pred = np.full_like(y, y.mean())
baseline_mae = mean_absolute_error(y, baseline_pred)

model_mae = mean_absolute_error(y, y_pred)
model_r2 = r2_score(y, y_pred)
slope = model.coef_[0]
intercept = model.intercept_

df["예측 승률"] = y_pred.round(3)
df["오차(실제-예측)"] = (df["W/L%"] - df["예측 승률"]).round(3)
df["절대오차"] = df["오차(실제-예측)"].abs().round(3)

# 상단 제목
st.title("🏀 NBA에서 3점슛이 승률에 정말 영향을 미치는가?")
st.caption("데이터 출처: Basketball-Reference (2025-26 NBA 정규시즌 30개 팀, 각 82경기 기준 | 전처리: 요약 행 1행 제외 31행→30행, 결측치 0행)")

# 핵심 지표 요약 카드
col1, col2, col3, col4 = st.columns(4)
col1.metric("회귀식 기울기", f"{slope:+.4f}", help=f"{selected_x}가 1 증가할 때 승률 변화량")
col2.metric("결정계수 (R²)", f"{model_r2:.3f}", help="1에 가까울수록 해당 변수로 승률 변동을 잘 설명함")
col3.metric("모델 평균절대오차 (MAE)", f"{model_mae:.3f}", delta=f"{model_mae - baseline_mae:+.3f} (베이스라인 대비)", delta_color="inverse")
col4.metric("베이스라인 MAE (평균 예측)", f"{baseline_mae:.3f}", help="모든 팀 승률을 리그 평균(0.500)으로 찍었을 때 오차")

st.info(
    f"**선형 회귀식:** `예측 W/L% = {slope:.4f} × ({selected_x}) + {intercept:.4f}`  |  "
    f"표본 수: **30개 팀** (결측치 0행)"
)

# 탭 구성
tab1, tab2, tab3 = st.tabs([
    "1. 산점도 & 회귀선 분석",
    "2. 예외 팀(오차 큰 팀) 해석",
    "3. 다중 변수 비교 & 원본 데이터",
])

with tab1:
    st.subheader(f"{selected_x}와 정규시즌 승률(W/L%)의 관계")

    x_range = np.linspace(df[selected_x].min(), df[selected_x].max(), 100)
    line_y = slope * x_range + intercept

    fig = px.scatter(
        df,
        x=selected_x,
        y="W/L%",
        hover_name="Team",
        hover_data={
            selected_x: True,
            "W/L%": ":.3f",
            "예측 승률": ":.3f",
            "오차(실제-예측)": ":+.3f",
            "PS/G": ":.1f",
            "PA/G": ":.1f",
        },
        labels={selected_x: f"{selected_x} (독립변수)", "W/L%": "정규시즌 승률 W/L% (종속변수)"},
        height=540,
    )
    fig.update_traces(marker=dict(size=11, color="#1f77b4", line=dict(width=1, color="white")))

    # 회귀선 추가
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=line_y,
            mode="lines",
            name=f"선형 회귀선 (R²={model_r2:.3f})",
            line=dict(color="#e63946", width=3),
        )
    )

    # 팀 이름 주석 표시
    for _, row in df.iterrows():
        if show_labels or row["Team"] in highlight_teams:
            fig.add_annotation(
                x=row[selected_x],
                y=row["W/L%"],
                text=row["Team"],
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-18,
                font=dict(size=11, color="#222222"),
            )

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("회귀선 예측과 실제 승률 차이가 큰 팀 (오차 분석)")
    st.write("양수(+)는 3점슛(또는 선택 변수) 대비 실제 승률이 훨씬 높은 팀, 음수(-)는 실제 승률이 훨씬 낮은 팀입니다.")

    sorted_err = df.sort_values("오차(실제-예측)", ascending=True)
    fig_err = px.bar(
        sorted_err,
        x="오차(실제-예측)",
        y="Team",
        orientation="h",
        color="오차(실제-예측)",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        hover_data=["3P", "W/L%", "예측 승률", "FG%", "PS/G", "PA/G"],
        height=680,
    )
    st.plotly_chart(fig_err, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**예측보다 실제 승률이 높았던 상위 5팀**")
        st.dataframe(
            df.sort_values("오차(실제-예측)", ascending=False)[["Team", "3P", "W/L%", "예측 승률", "오차(실제-예측)", "PA/G"]].head(5),
            hide_index=True,
            use_container_width=True,
        )
    with c2:
        st.markdown("**예측보다 실제 승률이 낮았던 하위 5팀**")
        st.dataframe(
            df.sort_values("오차(실제-예측)", ascending=True)[["Team", "3P", "W/L%", "예측 승률", "오차(실제-예측)", "PA/G"]].head(5),
            hide_index=True,
            use_container_width=True,
        )

with tab3:
    st.subheader("단일 변수(3P) vs 보조 변수 포함 다중 회귀 비교")
    feature_sets = {
        "기본 모델: 3P 단독": ["3P"],
        "3점슛 볼륨+효율: 3P + 3PA": ["3P", "3PA"],
        "공격 효율 추가: 3P + FG%": ["3P", "FG%"],
        "공격·수비 종합: 3P + PS/G + PA/G": ["3P", "PS/G", "PA/G"],
    }

    comp_rows = []
    for name, cols in feature_sets.items():
        m = LinearRegression().fit(df[cols], y)
        pred = m.predict(df[cols])
        comp_rows.append({
            "모델 구성": name,
            "사용 변수": ", ".join(cols),
            "결정계수 (R²)": round(r2_score(y, pred), 3),
            "평균절대오차 (MAE)": round(mean_absolute_error(y, pred), 3),
        })

    st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

    st.subheader("30개 팀 전처리 완료 데이터 표")
    st.dataframe(df, hide_index=True, use_container_width=True)

