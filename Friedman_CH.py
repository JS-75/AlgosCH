import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import warnings

def friedman_test(csv_file, start_col, end_col, output_file, comparisons_csv_file):
    """
    Realiza el test de Friedman y las comparaciones post-hoc de Nemenyi
    """
    
    # Leer el archivo CSV con la codificación apropiada
    try:
        data = pd.read_csv(csv_file, encoding='latin1')
    except:
        try:
            data = pd.read_csv(csv_file, encoding='iso-8859-1')
        except:
            import chardet
            with open(csv_file, 'rb') as file:
                raw_data = file.read()
            result = chardet.detect(raw_data)
            data = pd.read_csv(csv_file, encoding=result['encoding'])
    
    # Extraer las variables de interés
    variables = data.iloc[:, start_col:end_col+1]
    
    # Obtener número de evaluaciones y pacientes
    unique_evals = data['evaluacion'].unique()
    num_evals = len(unique_evals)
    num_patients = len(data['paciente'].unique())
    
    # Crear archivo para guardar resultados
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('RESULTADOS DEL TEST DE FRIEDMAN\n\n')
    
    # Crear DataFrame para almacenar todas las comparaciones múltiples
    all_comparisons = pd.DataFrame()
    
    # Realizar test de Friedman para cada variable
    for var_name in variables.columns:
        try:
            # Obtener los datos y verificar valores faltantes
            current_data = variables[var_name]
            if current_data.isnull().any():
                print(f"Advertencia: Se encontraron valores faltantes en {var_name}")
                continue
            
            # Reorganizar datos en un DataFrame pivot
            pivot_data = data.pivot(index='paciente', 
                                  columns='evaluacion', 
                                  values=var_name)
            
            # Convertir a matriz numpy
            X = pivot_data.values
            
            # Verificar que los datos son numéricos
            if not np.issubdtype(X.dtype, np.number):
                print(f"Advertencia: Los datos en {var_name} no son numéricos")
                continue
                
            # Verificar que hay suficiente variabilidad en los datos
            if np.all(X == X[0,0]):
                print(f"Advertencia: No hay variabilidad en los datos de {var_name}")
                continue
            
            print(X)
            # Realizar test de Friedman
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                statistic, p_value = stats.friedmanchisquare(*[X[:, i] for i in range(X.shape[1])])
            
            # Verificar resultados válidos
            if np.isnan(statistic) or np.isnan(p_value):
                print(f"Advertencia: Resultados no válidos para {var_name}")
                continue
            
            # Realizar test de Nemenyi
            nemenyi_results = sp.posthoc_nemenyi_friedman(X)
            
            # Escribir resultados en archivo
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f'=== Variable: {var_name} ===\n')
                f.write(f'Chi-square: {statistic:.4f}\n')
                f.write(f'Grados de libertad: {num_evals-1}\n')
                f.write(f'p-valor: {p_value:.4f}\n\n')
                
                f.write('Comparaciones múltiples (Test de Nemenyi):\n')
                f.write(nemenyi_results.to_string())
                f.write('\n\n')
            
            # Preparar resultados de Nemenyi para el CSV
            comparisons = []
            for i in range(num_evals):
                for j in range(i+1, num_evals):
                    comparisons.append({
                        'Variable': var_name,
                        'Grupo1': unique_evals[i],
                        'Grupo2': unique_evals[j],
                        'p_valor': nemenyi_results.iloc[i,j],
                    })
            
            # Agregar al DataFrame principal
            current_comparisons = pd.DataFrame(comparisons)
            all_comparisons = pd.concat([all_comparisons, current_comparisons], ignore_index=True)
            
        except Exception as e:
            print(f"Error procesando {var_name}: {str(e)}")
    
    # Guardar todas las comparaciones en un archivo CSV si hay resultados
    if not all_comparisons.empty:
        all_comparisons.to_csv(comparisons_csv_file, index=False, encoding='utf-8')
    else:
        print("No se generaron comparaciones válidas")


# Ejemplo de uso
if __name__ == "__main__":
    friedman_test(
        csv_file='C:/DATOS_CLINICOS/CH_R.csv',
        start_col=2,  # Columna de inicio de datos
        end_col=15,    # Columna final de los datos
        output_file='C:/DATOS_CLINICOS/resultados_friedman_CH.txt',
        comparisons_csv_file='C:/DATOS_CLINICOS/resultados_friedman_CH.csv'
    )