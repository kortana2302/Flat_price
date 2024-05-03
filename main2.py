import pandas as pd
import streamlit as st
import mpld3
import streamlit.components.v1 as components
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.preprocessing import Normalizer
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.title('Предсказание цены квартиры в Польше')

data = pd.read_csv('data (1).csv')

col_unique = {}
col_mode = {}
for col in data.columns:
    col_unique[col] = list(data[col].unique())
    if data[col].dtype == object:
        col_mode[col] = data[col].mode()[0]
    else:
        col_mode[col] = data[col].median()

col1, col2 = st.columns(2)
city = col1.selectbox(label = 'Название города', options = ['Выберите'] + col_unique['city'])
type_ = col1.selectbox(label = 'Тип квартиры', options = ['Выберите'] + col_unique['type'])
squareMeters = col1.number_input('Площадь квратиры', min_value=10, max_value=500)
rooms = col1.number_input('Кол-во комнат', min_value=1, max_value=10)
floor = col1.number_input('Этаж', min_value=1, max_value=80)
floorCount = col1.number_input('Кол-во этажей', min_value=1, max_value=80)
buildYear = col1.number_input('Дата постройки', min_value=1900, max_value=2024)
centreDistance = col1.number_input('Расстояние от центра города в км', min_value=0, max_value=200)
poiCount = col1.number_input('Количество достопримечательностей в радиусе 500 м', min_value=0, max_value=200)
ownership = col1.selectbox(label = 'Тип собственности на недвижимость', options = ['Выберите'] + col_unique['ownership'])
buildingMaterial = col1.selectbox(label = 'Материал постройки', options = ['Выберите'] + col_unique['buildingMaterial'])
condition = col1.selectbox(label = 'Состояние квартиры', options = ['Выберите'] + col_unique['condition'])
hasParkingSpace = col1.selectbox(label = 'Наличие парковки', options = ['Выберите'] + col_unique['hasParkingSpace'])
hasBalcony = col1.selectbox(label = 'Наличие балкона', options = ['Выберите'] + col_unique['hasBalcony'])
hasElevator = col1.selectbox(label = 'Наличие лифта', options = ['Выберите'] + col_unique['hasElevator'])
hasSecurity = col1.selectbox(label = 'Наличие охраны', options = ['Выберите'] + col_unique['hasSecurity'])
hasStorageRoom = col1.selectbox(label = 'Наличие кладовой', options = ['Выберите'] + col_unique['hasStorageRoom'])

df = pd.DataFrame(columns = ['city', 'type', 'squareMeters', 'rooms', 'floor', 'floorCount',
       'buildYear', 'latitude', 'longitude', 'centreDistance', 'poiCount',
       'schoolDistance', 'clinicDistance', 'postOfficeDistance',
       'kindergartenDistance', 'restaurantDistance', 'collegeDistance',
       'ownership', 'buildingMaterial', 'condition', 'hasParkingSpace',
       'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom'])

values_row = [city, type_, squareMeters, rooms, floor, floorCount,
       buildYear,centreDistance, poiCount, ownership, buildingMaterial, condition, hasParkingSpace,
       hasBalcony, hasElevator, hasSecurity, hasStorageRoom]
row_names = ['city', 'type', 'squareMeters', 'rooms', 'floor', 'floorCount',
       'buildYear', 'centreDistance', 'poiCount',
       'ownership', 'buildingMaterial', 'condition', 'hasParkingSpace',
       'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
dict_values = dict(zip(row_names, values_row))

row = []
for col in df.columns:
    if col in dict_values.keys():
        if dict_values[col] == 'Выберите':
            row.append(col_mode[col])
        else:
            row.append(dict_values[col])
    else:
        row.append(col_mode[col])
df.loc[len(df)] = row     

if st.button('Расчет') and len(df)>0:    
    col_numeric = data.columns[(data.dtypes==float) | (data.dtypes==int)]
    
    data = data.drop('pharmacyDistance', axis = 1)

    col_with_na = data.columns[data.isna().sum(axis = 0)>0]
    for col in col_with_na:
      if data[col].dtype == object:
        data[col] = data[col].fillna(data[col].mode()[0])
      else:
        data[col] = data[col].fillna(data[col].median())

    for col in ['type', 'ownership', 'buildingMaterial', 'condition']:
      ord_enc = OrdinalEncoder().fit(data[[col]])
      df[col] = ord_enc.transform(df[[col]])
      data[col] = ord_enc.transform(data[[col]])
    
    data_new = pd.get_dummies(data['city'])
    data[data_new.columns] = data_new
    data = data.drop('city', axis = 1)

    df[data_new.columns] = 0
    df[df['city']] = 1
    df = df.drop('city', axis = 1)
    
    for col in ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity','hasStorageRoom']:
      bn = LabelEncoder().fit(data[[col]])
      df[col] = bn.transform(df[[col]])
      data[col] = bn.transform(data[[col]])
    
    
    col_numeric = list(col_numeric)
    col_numeric.remove('pharmacyDistance')
    X_data = data.drop('price', axis = 1)
    transformer = MinMaxScaler().fit(X_data[col_numeric])
    X_data[col_numeric] = transformer.transform(X_data[col_numeric])

    y = data['price'].values
    transformer2 = StandardScaler().fit(y.reshape(-1, 1))
    y = transformer2.transform(y.reshape(-1,1))
    
    df[col_numeric] = transformer.transform(df[col_numeric])
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=10,
        learning_rate=0.01,
        n_estimators=1000)
    xgb_model.fit(X_data, y)
    y_pred = xgb_model.predict(df)
    y_pred = transformer2.inverse_transform(y_pred.reshape(-1,1))
    
    s = 'Предполагаемая цена жилья ' + str(int(y_pred[0][0]))
    
    col2.write(s)
