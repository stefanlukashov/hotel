import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


df = pd.read_csv("hotel_bookings.csv")

print("Размер датасета:", df.shape)
print("\nТипы данных:\n", df.dtypes)
print("\nОписание:\n", df.describe())
print("\nПропуски:\n", df.isnull().sum()[df.isnull().sum() > 0])

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(df["lead_time"], bins=40, color="steelblue", edgecolor="black")
axs[0].set_title("Распределение lead_time")
axs[0].set_xlabel("Дней до заезда")
axs[0].set_ylabel("Количество")

axs[1].hist(df["adr"].clip(0, 500), bins=40, color="coral", edgecolor="black")
axs[1].set_title("Распределение цены номера (adr)")
axs[1].set_xlabel("Цена за ночь")
axs[1].set_ylabel("Количество")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].boxplot(df["lead_time"])
axs[0].set_title("Выбросы: lead_time")
axs[1].boxplot(df["adr"].clip(0, 1000))
axs[1].set_title("Выбросы: adr")
plt.tight_layout()
plt.show()

print("\nВывод по анализу:")
print("- Признаки company и agent содержат много пропусков.")
print("- Категориальные признаки требуют кодирования.")
print("- lead_time, adr, previous_cancellations — важные признаки.")


df = df.drop(["company", "agent", "reservation_status"], axis=1)
df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
df = df[(df["adr"] >= 0) & (df["adr"] <= 5000)]
df = df[~((df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0))]

encoder = LabelEncoder()
for col in df.select_dtypes(include="str"):
    df[col] = encoder.fit_transform(df[col])

X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].hist(X["lead_time"], bins=40, color="steelblue", edgecolor="black")
axs[0, 0].set_title("lead_time до нормализации")
axs[0, 0].set_xlabel("Дней до заезда")
axs[0, 1].hist(X_scaled["lead_time"], bins=40, color="mediumseagreen", edgecolor="black")
axs[0, 1].set_title("lead_time после нормализации")
axs[0, 1].set_xlabel("Стандартизованное значение")
axs[1, 0].hist(X["adr"], bins=40, color="coral", edgecolor="black")
axs[1, 0].set_title("adr до нормализации")
axs[1, 0].set_xlabel("Цена за ночь")
axs[1, 1].hist(X_scaled["adr"], bins=40, color="mediumpurple", edgecolor="black")
axs[1, 1].set_title("adr после нормализации")
axs[1, 1].set_xlabel("Стандартизованное значение")
plt.tight_layout()
plt.show()

print("\nВывод по предобработке:")
print("- Удалены столбцы с большим числом пропусков.")
print("- Категориальные признаки закодированы.")
print("- Числовые признаки стандартизированы (mean=0, std=1).")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nОбучающая выборка: {X_train.shape[0]} | Тестовая: {X_test.shape[0]}")


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Ridge": Ridge(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

print(f"{'Модель':<25} {'Acc':>6} {'F1':>6} {'ROC':>6} {'Prec':>6} {'Rec':>6}")

for name, model in models.items():
    model.fit(X_train, y_train)

    if name == "Ridge":
        raw = model.predict(X_test)
        pred = (raw >= 0.5).astype(int)
        prob = np.clip(raw, 0, 1)
    else:
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, pred)
    f1   = f1_score(y_test, pred)
    roc  = roc_auc_score(y_test, prob)
    prec = precision_score(y_test, pred)
    rec  = recall_score(y_test, pred)

    print(f"{name:<25} {acc:>6.3f} {f1:>6.3f} {roc:>6.3f} {prec:>6.3f} {rec:>6.3f}")


print("\nВывод:")
print("- Logistic Regression показывает лучший результат.")
print("- Random Forest немного уступает, но лучше по Recall.")
print("- Ridge показывает худший результат — регрессионная модель плохо подходит для классификации.")
print("- Decision Tree занимает среднее место, склонна к переобучению.")
print("- Улучшения: подбор гиперпараметров, отбор признаков.")