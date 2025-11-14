# Install dependencies as needed:
# pip install kagglehub pandas
import kagglehub
import pandas as pd
import os

# Step 1: 데이터셋 다운로드
# print("데이터셋 다운로드 중...")
path = kagglehub.dataset_download("nathansmallcalder/lol-match-history-and-summoner-data-80k-matches")
# print(f"데이터셋 경로: {path}")

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

print(f"\n첫 번째 CSV 파일 로드: {csv_files[0]}")
df_0 = pd.read_csv(os.path.join(path, csv_files[0]))

# Step 4: 데이터 미리보기
print(f"\n데이터 크기: {df_0.shape}")
print("\nFirst 5 records:")
print(df_0.head())


df_1 = pd.read_csv(os.path.join(path, csv_files[1]))
print(f"\n두 번째 CSV 파일 로드: {csv_files[1]}")
print(f"\n데이터 크기: {df_1.shape}")
print("\nFirst 5 records:")
print(df_1.head())

df_2 = pd.read_csv(os.path.join(path, csv_files[2]))
print(f"\n세 번째 CSV 파일 로드: {csv_files[2]}")
print(f"\n데이터 크기: {df_2.shape}")
print("\nFirst 5 records:")
print(df_2.head())

df_3 = pd.read_csv(os.path.join(path, csv_files[3]))
print(f"\n네 번째 CSV 파일 로드: {csv_files[3]}")
print(f"\n데이터 크기: {df_3.shape}")
print("\nFirst 5 records:")
print(df_3.head())

df_4 = pd.read_csv(os.path.join(path, csv_files[4]))
print(f"\n다섯 번째 CSV 파일 로드: {csv_files[4]}")
print(f"\n데이터 크기: {df_4.shape}")
print("\nFirst 5 records:")
print(df_4.head())

df_5 = pd.read_csv(os.path.join(path, csv_files[5]))
print(f"\n여섯 번째 CSV 파일 로드: {csv_files[5]}")
print(f"\n데이터 크기: {df_5.shape}")
print("\nFirst 5 records:")
print(df_5.head())

df_6 = pd.read_csv(os.path.join(path, csv_files[6]))
print(f"\n일곱 번째 CSV 파일 로드: {csv_files[6]}")
print(f"\n데이터 크기: {df_6.shape}")
print("\nFirst 5 records:")
print(df_6.head())
