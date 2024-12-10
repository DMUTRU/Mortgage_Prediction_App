import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import datetime

# Установка заголовка приложения
st.title("Приложение для анализа и прогнозирования ипотечного рынка")

# Загрузка данных
st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите файл с данными (Excel или CSV)", type=["xlsx", "csv"])
data = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        st.sidebar.success("Файл успешно загружен!")
        st.write("Загруженные данные:")
        st.dataframe(data)
    except Exception as e:
        st.sidebar.error(f"Ошибка загрузки файла: {e}")

# Навигация
st.sidebar.header("Навигация")
page = st.sidebar.radio("Перейти к разделу", ["Анализ данных", "Планирование", "Сравнение моделей"])

# Раздел: Анализ данных
if page == "Анализ данных" and data is not None:
    st.header("Анализ данных")
    st.subheader("Описательная статистика")
    st.write(data.describe())
    st.subheader("Графики распределения данных")
    column = st.selectbox("Выберите переменную для анализа", data.columns)
    fig = px.histogram(data, x=column, nbins=20, title=f"Гистограмма для переменной: {column}")
    st.plotly_chart(fig)

# Раздел: Планирование
if page == "Планирование" and data is not None:
    st.header("Планирование")
    
    # Выбор целевой переменной и признаков
    target = st.selectbox("Выберите целевую переменную", data.columns)
    features = st.multiselect("Выберите признаки для прогнозирования", data.columns.drop(target))
    model_choice = st.selectbox("Выберите модель для прогнозирования", ["Линейная регрессия", "Random Forest", "ARIMA"])

    if target and features:
        X = data[features]
        y = data[target]

        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Линейная регрессия":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_choice == "Random Forest":
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_choice == "ARIMA":
            target_series = data[target]
            train_size = int(len(target_series) * 0.8)
            train, test = target_series[:train_size], target_series[train_size:]
            arima_model = ARIMA(train, order=(1, 1, 1))
            arima_result = arima_model.fit()
            forecast = arima_result.forecast(steps=len(test))
            y_pred = forecast
            y_test = test

        # Расчёт метрик
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
        st.write(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
        st.write(f"Коэффициент детерминации (R²): {r2:.2f}")

        # График прогнозируемых и фактических данных
        fig = px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': "Фактические значения", 'y': "Прогнозируемые значения"},
            title=f"Фактические vs Прогнозируемые значения ({model_choice})"
        )
        fig.add_scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name="Идеальное предсказание")
        st.plotly_chart(fig)

# Раздел: Сравнение моделей
if page == "Сравнение моделей" and data is not None:
    st.header("Сравнение моделей")
    
    target = st.selectbox("Выберите целевую переменную для моделей", data.columns)
    features = st.multiselect("Выберите признаки для сравнения моделей", data.columns.drop(target))

    if target and features:
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Линейная регрессия": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
        }

        # Добавление ARIMA
        target_series = data[target]
        train_size = int(len(target_series) * 0.8)
        train, test = target_series[:train_size], target_series[train_size:]
        arima_model = ARIMA(train, order=(1, 1, 1))
        arima_result = arima_model.fit()
        forecast = arima_result.forecast(steps=len(test))
        arima_metrics = {
            "MSE": mean_squared_error(test, forecast),
            "MAE": mean_absolute_error(test, forecast),
            "R²": 1 - (sum((test - forecast) ** 2) / sum((test - np.mean(test)) ** 2))
        }

        results = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[model_name] = {"MSE": mse, "MAE": mae, "R²": r2}

        # Добавление ARIMA в результаты
        results["ARIMA"] = arima_metrics
        st.write(pd.DataFrame(results).T)
