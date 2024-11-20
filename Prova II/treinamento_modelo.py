from importacao_insumos import importa_dataset_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

def pre_processamento_modelo(df):
    # Separar features e o alvo
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Normalizar as features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, scaler

def treinamento_modelo(X_train, y_train):
    # Treinar um modelo de classificação
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def avaliacao_modelo(model, X_test, y_test):
    # Realizar predições com o conjunto de teste
    y_pred = model.predict(X_test)
    
    # Calcular a acurácia
    acuracia = accuracy_score(y_test, y_pred)
    
    print(f"Acurácia do modelo: {acuracia * 100:.2f}%")
    return acuracia

def predicao_interativa(model, scaler):
    # Coletar inputs do usuário
    print("Por favor, insira os seguintes dados:")
    Pregnancies = float(input("Número de gravidezes: "))
    Glucose = float(input("Nível de glicose: "))
    BloodPressure = float(input("Pressão arterial: "))
    SkinThickness = float(input("Espessura da pele: "))
    Insulin = float(input("Nível de insulina: "))
    BMI = float(input("Índice de massa corporal (BMI): "))
    DiabetesPedigreeFunction = float(input("Função de pedigree diabético: "))
    Age = float(input("Idade: "))
    
    # Montar a entrada para o modelo
    user_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    user_data_scaled = scaler.transform(user_data)  # Normalizar os dados
    
    # Fazer a predição
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)
    
    # Retornar os resultados
    if prediction[0] == 1:
        print(f"\n**Resultado**: Alta probabilidade de diabetes (Chance: {probability[0][1] * 100:.2f}%)")
    else:
        print(f"\n**Resultado**: Baixa probabilidade de diabetes (Chance: {probability[0][0] * 100:.2f}%)")

def run():
    # Chamar a função para carregar o dataset
    df = importa_dataset_diabetes()

    # Chamar a função de pré-processamento
    X_train, X_test, y_train, y_test, scaler = pre_processamento_modelo(df)

    # Chamar a função de treinamento do modelo
    model = treinamento_modelo(X_train, y_train)

    # Avaliar o modelo e imprimir a acurácia
    avaliacao_modelo(model, X_test, y_test)

    # Chamar a função interativa para predição
    predicao_interativa(model, scaler)

run()