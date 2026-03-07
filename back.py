import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

df_fullclean = pd.read_csv('ultimate_clean.csv')
#print(df_fullclean.head())

# Combine similar rare classes into broader categories
df_fullclean['EVENT_TYPE'] = df_fullclean['EVENT_TYPE'].replace({
    'Marine Hail': 'Marine Event',
    'Marine High Wind': 'Marine Event',
    'Marine Strong Wind': 'Marine Event',
    'Marine Thunderstorm Wind': 'Marine Event',
    'Marine Dense Fog': 'Marine Event',
    'Marine Tropical Storm': 'Marine Event',
    'Marine Dense Storm': 'Marine Event',
    'Rip Current': 'Coastal Hazard',
    'High Surf': 'Coastal Hazard',
    'Sneakerwave': 'Coastal Hazard',
    'Coastal Flood': 'Coastal Hazard',
    'Cold/Wind Chill': 'Extreme Cold',
    'Extreme Cold/Wind Chill': 'Extreme Cold'
})

storm_categories = df_fullclean['EVENT_TYPE'].unique().tolist()

# Print them out clearly, one by one
for storm in sorted(storm_categories):
    print(storm)

# plt.figure(figsize=(15, 10))
# ax = sns.countplot(y='EVENT_TYPE', data=df_fullclean, order = df_fullclean['EVENT_TYPE'].value_counts().index)
# ax.tick_params(axis='y', labelsize=9)
# ax.set_xlabel("Storms in 2025")
# plt.show()

df_fullclean['BEGIN_YEARMONTH'] = df_fullclean['BEGIN_YEARMONTH'].astype(str)
df_fullclean['BEGIN_YEAR'] = df_fullclean['BEGIN_YEARMONTH'].str[0:4]
df_fullclean['BEGIN_MONTH'] = df_fullclean['BEGIN_YEARMONTH'].str[5:6]
df_fullclean['BEGIN_YEAR'] = df_fullclean['BEGIN_YEAR'].astype(int)
df_fullclean['BEGIN_MONTH'] = df_fullclean['BEGIN_MONTH'].astype(int)

def assign_region(row):
    lat = row['lat']
    lon = row['lng']
    
    if lat > 35 and lon < -90 and lon > -110:
        return 'CENTRAL'
    elif lat > 35 and lon > -90:
        return 'EAST'
    elif lat < 35:
        return 'SOUTH'
    elif lat > 35 and lon < -110:
        return 'WEST'
    else:
        return 'UNKNOWN'

# Apply the function to every row (axis=1)
df_fullclean['REGION'] = df_fullclean.apply(assign_region, axis=1)

#Psuedo-negatives
num_negatives = int(len(df_fullclean)*0.1)

#Generate random data within the same range as your real data
negatives = pd.DataFrame({
    'lat': np.random.uniform(25, 50, num_negatives),  # Approx US Latitudes
    'lng': np.random.uniform(-125, -65, num_negatives), # Approx US Longitudes
    'BEGIN_MONTH': np.random.randint(1, 13, num_negatives),
    'BEGIN_DAY': np.random.randint(1, 29, num_negatives), # Safety cap at 28 to avoid Feb 30th errors
    'EVENT_TYPE': 'No Storm', # The label for these rows
    'DAMAGE_PROPERTY': 0.0
})

negatives['REGION'] = negatives.apply(assign_region, axis=1)

#Merging cases + pseudo-negatives
df_combined = pd.concat([df_fullclean, negatives], ignore_index=True)

print(f"Original size: {len(df_fullclean)}")
print(f"New size: {len(df_combined)}")

#Creating dataset for dmg

#Training event
X = df_combined[['lat', 'lng', 'BEGIN_MONTH', 'BEGIN_DAY']]
y = df_combined['EVENT_TYPE'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model = RandomForestClassifier(
    n_estimators=50,        # Use 50 trees instead of the default 100
    max_depth=15,           # Do not let any tree grow deeper than 15 levels
    min_samples_leaf=5,     # A branch must have at least 5 storms to be saved
    random_state=42, 
    class_weight='balanced'
)

model.fit(X_train, y_train)

#Training prop dmg
X_dmg = df_combined[['lat', 'lng', 'BEGIN_MONTH', 'BEGIN_DAY', 'EVENT_TYPE']]
X_dmg_encoded = pd.get_dummies(X_dmg, columns=['EVENT_TYPE'])
y_dmg = df_combined['DAMAGE_PROPERTY'] 
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dmg_encoded, y_dmg, test_size=0.2)
dmg_model = RandomForestRegressor(random_state=42)
dmg_model = RandomForestRegressor(
    n_estimators=50, 
    max_depth=15, 
    min_samples_leaf=5, 
    random_state=42
)

dmg_model.fit(X_train_d, y_train_d)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Houston: 30, -95
# LA: 34, -118
# Chicago: 41, -87
# Seattle 

user_lat = 41
user_lon = -87
user_month = 1
user_day = 15

try:
    #month_code = le_month.transform([user_month])[0]
    user_input = [[user_lat, user_lon, user_month, user_day]]

    # A. Get the Predicted Storm Type
    prediction = model.predict(user_input)

    dmg_user_input = [[user_lat, user_lon, user_month, user_day, prediction[0]]]
    pred_dmg = dmg_model.predict(dmg_user_input)
    
    # B. Get the Probability (Confidence)
    probabilities = model.predict_proba(user_input)
    
    print(f"User Location: {user_lat}, {user_lon} in month: {user_month} and day: {user_day}")
    print(f"Predicted Most Likely Storm: {prediction[0]}")
    print(pred_dmg)
    
    # Show probability for each possibility
    print("\nProbability Breakdown:")
    classes = model.classes_
    probs = probabilities[0]
    for storm_name, prob in zip(classes, probs):
        if prob > 0:
            print(f" - {storm_name}: {prob:.1%}")
                     
except ValueError:
    print("Error: Month name not recognized (check spelling).")

joblib.dump(model, 'storm_model.pkl')
joblib.dump(dmg_model, 'damage_model.pkl')
print("Model saved as storm_model.pkl!")