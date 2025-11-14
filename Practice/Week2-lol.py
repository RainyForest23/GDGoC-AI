import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ============================================================
# LoL 매치 데이터셋 구성 (Kaggle Dataset)
# Dataset: lol-match-history-and-summoner-data-80k-matches
# ============================================================

# ------------------------------------------------------------
# 1. 데이터셋 다운로드
# ------------------------------------------------------------
# Install dependencies as needed:
# pip install kagglehub pandas
import kagglehub
import pandas as pd
import os

# Step 1: 데이터셋 다운로드
path = kagglehub.dataset_download("nathansmallcalder/lol-match-history-and-summoner-data-80k-matches")

# Step 2: 다운로드된 파일 목록 확인
print("\n사용 가능한 파일:")
files = os.listdir(path)
for f in files:
    print(f"  - {f}")

# Step 3: CSV 파일 찾기
csv_files = [f for f in files if f.endswith('.csv')]
if not csv_files:
    print("\nCSV 파일을 찾을 수 없습니다.")
    exit()
# ------------------------------------------------------------



# ============================================================
# 승패 예측 (지도학습 - 분류)
# ============================================================
#    - 데이터: MatchStatsTbl
#    - 특성: kills, deaths, assists, MinionsKilled, TotalGold,
#            DmgDealt, DragonKills, BaronKills, visionScore
#    - 타겟: Win (0/1)
#    - 모델: Logistic Regression, Decision Tree, Random Forest
# ============================================================



# 1단계: 데이터 로드 및 확인
print("=" * 60)
print("1단계: 데이터 로드 및 확인")
print("=" * 60)

# 데이터 경로
data_path = "/Users/rainyforest/.cache/kagglehub/datasets/nathansmallcalder/lol-match-history-and-summoner-data-80k-matches/versions/1"

# MatchStatsTbl 로드
df = pd.read_csv(f"{data_path}/MatchStatsTbl.csv")

print(f"\n✓ 데이터 로드 완료!")
print(f"  - 데이터 크기: {df.shape}")
print(f"  - 총 행: {df.shape[0]:,}개")
print(f"  - 총 열: {df.shape[1]}개")

print(f"\n✓ 컬럼 목록:")
print(f"  {df.columns.tolist()}")

print(f"\n✓ 처음 5개 행:")
print(df.head())


# 2단계: 데이터 탐색 (EDA)
print("\n" + "=" * 60)
print("2단계: 데이터 탐색 (EDA)")
print("=" * 60)

# 2-1. 타겟 변수 분포 확인
print(f"\n[2-1] 타겟 변수(Win) 분포:")
win_counts = df['Win'].value_counts()
win_ratio = df['Win'].value_counts(normalize=True) * 100

print(f"\n  패배(0): {win_counts[0]:,}개 ({win_ratio[0]:.2f}%)")
print(f"  승리(1): {win_counts[1]:,}개 ({win_ratio[1]:.2f}%)")

# 시각화
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Win')
plt.title('승패 분포', fontsize=14)
plt.xlabel('승패 (0: 패배, 1: 승리)')
plt.ylabel('경기 수')
plt.xticks([0, 1], ['패배', '승리'])
for i, v in enumerate(win_counts):
    plt.text(i, v + 500, f'{v:,}\n({win_ratio[i]:.1f}%)', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('/Users/rainyforest/Desktop/github/GDGoC/Practice/win_distribution.png', dpi=150)
print(f"\n  ✓ 그래프 저장: win_distribution.png")
plt.close()

# 2-2. 결측치 확인
print(f"\n[2-2] 결측치 확인:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    '결측치 수': missing,
    '비율(%)': missing_pct
})
missing_info = missing_df[missing_df['결측치 수'] > 0].sort_values('결측치 수', ascending=False)

if len(missing_info) > 0:
    print(missing_info)
else:
    print("  ✓ 결측치 없음!")

# 2-3. 기술 통계
print(f"\n[2-3] 주요 특성 기술 통계:")
key_stats = ['kills', 'deaths', 'assists', 'MinionsKilled', 'TotalGold',
             'DmgDealt', 'DragonKills', 'BaronKills', 'visionScore']
print(df[key_stats].describe())

# 2-4. 승패별 통계 비교
print(f"\n[2-4] 승패별 평균 통계 비교:")
win_stats = df.groupby('Win')[key_stats].mean()
print(win_stats.T)

# 시각화
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(key_stats):
    sns.boxplot(data=df, x='Win', y=col, ax=axes[idx])
    axes[idx].set_title(f'{col} - 승패별 비교')
    axes[idx].set_xlabel('승패 (0: 패배, 1: 승리)')
    axes[idx].set_xticklabels(['패배', '승리'])

plt.tight_layout()
plt.savefig('/Users/rainyforest/Desktop/github/GDGoC/Practice/stats_by_win.png', dpi=150)
print(f"\n  ✓ 그래프 저장: stats_by_win.png")
plt.close()


# 3단계: 데이터 전처리
print("\n" + "=" * 60)
print("3단계: 데이터 전처리")
print("=" * 60)

# 3-1. 결측치 처리
print(f"\n[3-1] 결측치 처리:")
df_clean = df.copy()

# 결측치가 있는 컬럼은 중앙값으로 채우기
if missing.sum() > 0:
    imputer = SimpleImputer(strategy='median')
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    print(f"  ✓ 수치형 컬럼 결측치를 중앙값으로 대체")
else:
    print(f"  ✓ 결측치가 없어 건너뜀")

# 3-2. 특성 선택
print(f"\n[3-2] 특성 선택:")

# 사용할 특성 (게임 플레이 통계)
feature_cols = [
    'kills', 'deaths', 'assists',
    'MinionsKilled',
    'TotalGold',
    'DmgDealt', 'DmgTaken',
    'TurretDmgDealt',
    'DragonKills', 'BaronKills',
    'visionScore'
]

# 타겟 변수
target_col = 'Win'

print(f"  ✓ 선택한 특성 ({len(feature_cols)}개):")
for i, col in enumerate(feature_cols, 1):
    print(f"    {i:2d}. {col}")

print(f"\n  ✓ 타겟 변수: {target_col}")

# 3-3. X, y 분리
X = df_clean[feature_cols]
y = df_clean[target_col]

print(f"\n  ✓ 특성(X) 크기: {X.shape}")
print(f"  ✓ 타겟(y) 크기: {y.shape}")


# 4단계: 데이터 분할
print("\n" + "=" * 60)
print("4단계: 데이터 분할 (학습/테스트)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 승패 비율 유지
)

print(f"\n  ✓ 학습 데이터: {X_train.shape[0]:,}개 ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  ✓ 테스트 데이터: {X_test.shape[0]:,}개 ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\n  ✓ 학습 데이터 승패 비율:")
train_win_ratio = y_train.value_counts(normalize=True) * 100
print(f"    - 패배: {train_win_ratio[0]:.2f}%")
print(f"    - 승리: {train_win_ratio[1]:.2f}%")

print(f"\n  ✓ 테스트 데이터 승패 비율:")
test_win_ratio = y_test.value_counts(normalize=True) * 100
print(f"    - 패배: {test_win_ratio[0]:.2f}%")
print(f"    - 승리: {test_win_ratio[1]:.2f}%")


# 5단계: 특성 스케일링
print("\n" + "=" * 60)
print("5단계: 특성 스케일링 (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n  ✓ 스케일링 전 (학습 데이터):")
print(f"    - 평균: {X_train.mean().mean():.2f}")
print(f"    - 표준편차: {X_train.std().mean():.2f}")

print(f"\n  ✓ 스케일링 후 (학습 데이터):")
print(f"    - 평균: {X_train_scaled.mean():.6f}")
print(f"    - 표준편차: {X_train_scaled.std():.2f}")

print(f"\n  ✓ 스케일링 완료!")


# 6단계: 모델 학습
print("\n" + "=" * 60)
print("6단계: 모델 학습")
print("=" * 60)

# 6-1. Logistic Regression
print(f"\n[6-1] Logistic Regression 학습:")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
print(f"  ✓ 학습 완료!")

# 6-2. Decision Tree
print(f"\n[6-2] Decision Tree 학습:")
dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=100, random_state=42)
dt_model.fit(X_train, y_train)  # Decision Tree는 스케일링 불필요
print(f"  ✓ 학습 완료!")


# 7단계: 모델 평가
print("\n" + "=" * 60)
print("7단계: 모델 평가")
print("=" * 60)

# 7-1. Logistic Regression 평가
print(f"\n[7-1] Logistic Regression 평가:")

y_pred_lr = lr_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"\n  ✓ 정확도: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"\n  ✓ Confusion Matrix:")
print(f"              예측 패배  예측 승리")
print(f"    실제 패배   {cm_lr[0,0]:6d}    {cm_lr[0,1]:6d}   (TN: {cm_lr[0,0]}, FP: {cm_lr[0,1]})")
print(f"    실제 승리   {cm_lr[1,0]:6d}    {cm_lr[1,1]:6d}   (FN: {cm_lr[1,0]}, TP: {cm_lr[1,1]})")

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['패배', '승리'],
            yticklabels=['패배', '승리'])
plt.xlabel('예측 레이블')
plt.ylabel('실제 레이블')
plt.title(f'Logistic Regression - Confusion Matrix\n정확도: {accuracy_lr:.4f}')
plt.tight_layout()
plt.savefig('/Users/rainyforest/Desktop/github/GDGoC/Practice/cm_logistic.png', dpi=150)
print(f"\n  ✓ 그래프 저장: cm_logistic.png")
plt.close()

# Feature Importance (계수의 절대값)
feature_importance_lr = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n  ✓ 상위 5개 중요 특성:")
print(feature_importance_lr.head(5).to_string(index=False))

# 7-2. Decision Tree 평가
print(f"\n[7-2] Decision Tree 평가:")

y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print(f"\n  ✓ 정확도: {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(f"\n  ✓ Confusion Matrix:")
print(f"              예측 패배  예측 승리")
print(f"    실제 패배   {cm_dt[0,0]:6d}    {cm_dt[0,1]:6d}   (TN: {cm_dt[0,0]}, FP: {cm_dt[0,1]})")
print(f"    실제 승리   {cm_dt[1,0]:6d}    {cm_dt[1,1]:6d}   (FN: {cm_dt[1,0]}, TP: {cm_dt[1,1]})")

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens',
            xticklabels=['패배', '승리'],
            yticklabels=['패배', '승리'])
plt.xlabel('예측 레이블')
plt.ylabel('실제 레이블')
plt.title(f'Decision Tree - Confusion Matrix\n정확도: {accuracy_dt:.4f}')
plt.tight_layout()
plt.savefig('/Users/rainyforest/Desktop/github/GDGoC/Practice/cm_tree.png', dpi=150)
print(f"\n  ✓ 그래프 저장: cm_tree.png")
plt.close()

# Feature Importance
feature_importance_dt = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n  ✓ 상위 5개 중요 특성:")
print(feature_importance_dt.head(5).to_string(index=False))

# Decision Tree 시각화
plt.figure(figsize=(20, 12))
plot_tree(dt_model,
          max_depth=3,
          feature_names=feature_cols,
          class_names=['패배', '승리'],
          filled=True,
          fontsize=10)
plt.title('Decision Tree 구조 (상위 3레벨)', fontsize=16)
plt.tight_layout()
plt.savefig('/Users/rainyforest/Desktop/github/GDGoC/Practice/tree_structure.png', dpi=150)
print(f"\n  ✓ 그래프 저장: tree_structure.png")
plt.close()


# 8단계: 모델 비교
print("\n" + "=" * 60)
print("8단계: 모델 비교")
print("=" * 60)

comparison = pd.DataFrame({
    '모델': ['Logistic Regression', 'Decision Tree'],
    '정확도': [accuracy_lr, accuracy_dt],
    '정확도(%)': [accuracy_lr*100, accuracy_dt*100]
})

print(f"\n{comparison.to_string(index=False)}")

# 시각화
plt.figure(figsize=(10, 6))
bars = plt.bar(comparison['모델'], comparison['정확도(%)'], color=['#3498db', '#2ecc71'])
plt.ylim(0, 100)
plt.ylabel('정확도 (%)', fontsize=12)
plt.title('모델 정확도 비교', fontsize=14)
plt.grid(axis='y', alpha=0.3)

# 값 표시
for i, (bar, row) in enumerate(zip(bars, comparison.itertuples())):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{row[3]:.2f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/rainyforest/Desktop/github/GDGoC/Practice/model_comparison.png', dpi=150)
print(f"\n  ✓ 그래프 저장: model_comparison.png")
plt.close()


# 9단계: 결론
print("\n" + "=" * 60)
print("9단계: 결론 및 인사이트")
print("=" * 60)

print(f"\n✓ 최종 결과:")
print(f"  - Logistic Regression 정확도: {accuracy_lr*100:.2f}%")
print(f"  - Decision Tree 정확도: {accuracy_dt*100:.2f}%")

better_model = "Logistic Regression" if accuracy_lr > accuracy_dt else "Decision Tree"
print(f"\n  ⭐ 더 좋은 모델: {better_model}")

print(f"\n✓ 주요 발견 (Logistic Regression 기준):")
print(f"  상위 3개 중요 특성:")
for i, row in feature_importance_lr.head(3).iterrows():
    direction = "승리에 긍정적" if row['Coefficient'] > 0 else "패배와 관련"
    print(f"    {i+1}. {row['Feature']}: 계수 {row['Coefficient']:.4f} ({direction})")

print(f"\n✓ 주요 발견 (Decision Tree 기준):")
print(f"  상위 3개 중요 특성:")
for i, row in feature_importance_dt.head(3).iterrows():
    print(f"    {i+1}. {row['Feature']}: 중요도 {row['Importance']:.4f}")

print(f"\n✓ 개선 아이디어:")
print(f"  1. 특성 엔지니어링: KDA = (kills+assists)/(deaths+1) 추가")
print(f"  2. 더 복잡한 모델: Random Forest, XGBoost 시도")
print(f"  3. 하이퍼파라미터 튜닝: Grid Search 적용")
print(f"  4. 추가 데이터: 챔피언, 아이템, 룬 정보 활용")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)
print(f"\n생성된 파일:")
print(f"  - win_distribution.png")
print(f"  - stats_by_win.png")
print(f"  - cm_logistic.png")
print(f"  - cm_tree.png")
print(f"  - tree_structure.png")
print(f"  - model_comparison.png")

