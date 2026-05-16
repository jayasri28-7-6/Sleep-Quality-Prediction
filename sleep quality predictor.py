import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

plt.ion()

np.random.seed(42)
n = 100

sleep_duration = np.random.randint(4, 11, size=n)
screen_time = np.random.randint(0, 181, size=n)
exercise_duration = np.random.randint(0, 121, size=n)
stress_level = np.random.randint(0, 11, size=n)
caffeine_intake = np.random.choice(['None', 'Low', 'Moderate', 'High'], size=n)

sleep_quality = []
for s, sc, e, st, c in zip(sleep_duration, screen_time, exercise_duration, stress_level, caffeine_intake):
    score = 0
    if s >= 7: score += 2
    if sc <= 60: score += 1
    if e >= 30: score += 1
    if st <= 5: score += 1
    if c in ['None', 'Low']: score += 1

    if score <= 2:
        sleep_quality.append('Poor')
    elif score <= 4:
        sleep_quality.append('Average')
    else:
        sleep_quality.append('Good')

data = pd.DataFrame({
    "SleepDuration": sleep_duration,
    "ScreenTime": screen_time,
    "ExerciseDuration": exercise_duration,
    "StressLevel": stress_level,
    "CaffeineIntake": caffeine_intake,
    "SleepQuality": sleep_quality
})

le_caffeine = LabelEncoder()
le_sleep = LabelEncoder()

data["CaffeineIntake"] = le_caffeine.fit_transform(data["CaffeineIntake"])
data["SleepQuality"] = le_sleep.fit_transform(data["SleepQuality"])

X = data.drop("SleepQuality", axis=1)
y = data["SleepQuality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nSleep Quality Predictor")
sleep = float(input("Enter sleep duration (hours): "))
screen = int(input("Screen time before bed (minutes): "))
exercise = int(input("Exercise duration (minutes): "))
stress = int(input("Stress level (0–10): "))

print("\nCaffeine Intake:")
for idx, val in enumerate(le_caffeine.classes_):
    print(f"{idx} - {val}")
caffeine = int(input("Enter caffeine level: "))

user_data = np.array([[sleep, screen, exercise, stress, caffeine]])
prediction = model.predict(user_data)[0]
result = le_sleep.inverse_transform([prediction])[0]
print("\nPredicted Sleep Quality:", result)

sleep_score = 100
if sleep < 7: sleep_score -= 20
if screen > 90: sleep_score -= 15
if exercise < 30: sleep_score -= 15
if stress > 6: sleep_score -= 20
sleep_score = max(sleep_score, 0)
print("Sleep Health Score:", sleep_score, "/100")

print("\nSuggestions:")
if sleep < 7: print("- Increase sleep duration")
if screen > 90: print("- Reduce screen time before bed")
if exercise < 30: print("- Increase physical activity")
if stress > 6: print("- Practice relaxation techniques")

ideal_profile = [8, 30, 45, 2]

fig1 = plt.figure(figsize=(6, 4))
plt.bar(["Sleep", "Screen", "Exercise", "Stress"], [sleep, screen, exercise, stress], color=['#4CAF50','#FF5722','#03A9F4','#FFC107'])
plt.title("Current Sleep Factors")
plt.ylabel("Values")
plt.show()
input("Press Enter to see the suggested sleep improvement plan...")

fig2 = plt.figure(figsize=(7, 4))
plt.bar(["Sleep", "Screen", "Exercise", "Stress"], ideal_profile, color=['#4CAF50','#FF5722','#03A9F4','#FFC107'])
plt.title("Suggested Sleep Improvement Plan")
plt.ylabel("Recommended Values")
plt.show()
input("Press Enter to compare current vs suggested sleep factors...")

x = np.arange(4)
fig3 = plt.figure(figsize=(7,4))
plt.bar(x - 0.2, [sleep, screen, exercise, stress], width=0.4, label="Current", color='#2196F3')
plt.bar(x + 0.2, ideal_profile, width=0.4, label="Suggested", color='#FF9800')
plt.xticks(x, ["Sleep", "Screen", "Exercise", "Stress"])
plt.ylabel("Values")
plt.title("Current vs Suggested Sleep Factors")
plt.legend()
plt.show()

plt.ioff()
plt.show()
