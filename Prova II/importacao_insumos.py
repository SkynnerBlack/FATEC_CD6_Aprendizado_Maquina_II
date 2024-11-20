import pandas as pd
import kagglehub

def importa_dataset_diabetes():
    """
    Faz o download do dataset de diabetes do Kaggle e retorna um DataFrame do Pandas.
    
    Returns:
        pd.DataFrame: DataFrame contendo os dados do dataset.
    """
    # Faz o download da versão mais nova
    path = kagglehub.dataset_download("akshaydattatraykhare/diabetes-dataset")
    print("Path to dataset files:", path)

    # Carrega a tabela em formato Pandas Dataframe
    file_path = f"{path}/diabetes.csv"
    df = pd.read_csv(file_path)
    return df
