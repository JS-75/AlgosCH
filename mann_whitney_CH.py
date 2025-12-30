import pandas as pd
import numpy as np
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_comparison_plots(data1, data2, variables, evaluaciones, output_folder, 
                           group1_name='Grupo 1', group2_name='Grupo 2',
                           dpi=300, format='png'):
    """
    Crea box plots comparativos para cada variable
    
    Parámetros:
    data1, data2: DataFrames con los datos de cada grupo
    variables: lista de variables a graficar
    evaluaciones: lista de evaluaciones
    output_folder: carpeta donde guardar las imágenes
    group1_name, group2_name: nombres de los grupos para la leyenda
    dpi: resolución de la imagen (300 para publicación)
    format: formato de imagen ('png', 'pdf', 'svg', 'tiff')
    """
    
    # Crear carpeta de salida si no existe
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Configurar estilo para publicación científica
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    
    for var in variables:
        try:
            # Preparar datos para el gráfico
            plot_data = []
            medians_g1 = []
            medians_g2 = []
            eval_list = []
            
            for eval_num in evaluaciones:
                # Datos grupo 1
                datos_g1 = data1[data1['evaluacion'] == eval_num][var].dropna()
                # Datos grupo 2
                datos_g2 = data2[data2['evaluacion'] == eval_num][var].dropna()
                
                if len(datos_g1) > 0:
                    for val in datos_g1:
                        plot_data.append({
                            'Evaluacion': eval_num,
                            'Valor': val,
                            'Grupo': group1_name
                        })
                    medians_g1.append(datos_g1.median())
                else:
                    medians_g1.append(np.nan)
                
                if len(datos_g2) > 0:
                    for val in datos_g2:
                        plot_data.append({
                            'Evaluacion': eval_num,
                            'Valor': val,
                            'Grupo': group2_name
                        })
                    medians_g2.append(datos_g2.median())
                else:
                    medians_g2.append(np.nan)
                
                eval_list.append(eval_num)
            
            if not plot_data:
                print(f"No hay datos suficientes para graficar {var}")
                continue
            
            # Crear DataFrame para seaborn
            df_plot = pd.DataFrame(plot_data)
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Crear boxplot
            box_width = 0.35
            positions_g1 = np.array(range(len(evaluaciones))) - box_width/2
            positions_g2 = np.array(range(len(evaluaciones))) + box_width/2
            
            # Preparar datos para boxplot manual
            data_g1 = [data1[data1['evaluacion'] == e][var].dropna().values 
                      for e in evaluaciones]
            data_g2 = [data2[data2['evaluacion'] == e][var].dropna().values 
                      for e in evaluaciones]
            
            # Crear boxplots
            bp1 = ax.boxplot(data_g1, positions=positions_g1, widths=box_width,
                            patch_artist=True, 
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='darkblue', linewidth=2),
                            whiskerprops=dict(color='darkblue', linewidth=1.5),
                            capprops=dict(color='darkblue', linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor='lightblue', 
                                          markersize=5, alpha=0.5))
            
            bp2 = ax.boxplot(data_g2, positions=positions_g2, widths=box_width,
                            patch_artist=True,
                            boxprops=dict(facecolor='lightcoral', alpha=0.7),
                            medianprops=dict(color='darkred', linewidth=2),
                            whiskerprops=dict(color='darkred', linewidth=1.5),
                            capprops=dict(color='darkred', linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor='lightcoral', 
                                          markersize=5, alpha=0.5))
            
            # Líneas de medianas
            medians_g1_clean = np.array(medians_g1)
            medians_g2_clean = np.array(medians_g2)
            
            valid_idx_g1 = ~np.isnan(medians_g1_clean)
            valid_idx_g2 = ~np.isnan(medians_g2_clean)
            
            if np.sum(valid_idx_g1) > 1:
                ax.plot(positions_g1[valid_idx_g1], medians_g1_clean[valid_idx_g1], 
                       'o-', color='darkblue', linewidth=2, markersize=8, 
                       label=f'{group1_name} (mediana)', zorder=10)
            
            if np.sum(valid_idx_g2) > 1:
                ax.plot(positions_g2[valid_idx_g2], medians_g2_clean[valid_idx_g2], 
                       'o-', color='darkred', linewidth=2, markersize=8, 
                       label=f'{group2_name} (mediana)', zorder=10)
            
            # Líneas de tendencia (regresión lineal)
            if np.sum(valid_idx_g1) > 1:
                x_g1 = positions_g1[valid_idx_g1]
                y_g1 = medians_g1_clean[valid_idx_g1]
                z_g1 = np.polyfit(x_g1, y_g1, 1)
                p_g1 = np.poly1d(z_g1)
                ax.plot(positions_g1[valid_idx_g1], p_g1(positions_g1[valid_idx_g1]), 
                       '--', color='blue', linewidth=1.5, alpha=0.7, 
                       label=f'{group1_name} (tendencia)')
            
            if np.sum(valid_idx_g2) > 1:
                x_g2 = positions_g2[valid_idx_g2]
                y_g2 = medians_g2_clean[valid_idx_g2]
                z_g2 = np.polyfit(x_g2, y_g2, 1)
                p_g2 = np.poly1d(z_g2)
                ax.plot(positions_g2[valid_idx_g2], p_g2(positions_g2[valid_idx_g2]), 
                       '--', color='red', linewidth=1.5, alpha=0.7, 
                       label=f'{group2_name} (tendencia)')
            
            # Configurar ejes y etiquetas
            ax.set_xticks(range(len(evaluaciones)))
            ax.set_xticklabels([f'T{e}' for e in evaluaciones])
            ax.set_xlabel('Momento de evaluación', fontsize=12, fontweight='bold')
            ax.set_ylabel(var, fontsize=12, fontweight='bold')
            ax.set_title(f'Comparación de {var} entre grupos', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Leyenda
            ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar figura
            filename = f"{var.replace(' ', '_').replace('/', '_')}.{format}"
            filepath = Path(output_folder) / filename
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
            plt.close()
            
            print(f"Gráfico guardado: {filepath}")
            
        except Exception as e:
            print(f"Error al crear gráfico para {var}: {str(e)}")
            plt.close()

def mann_whitney_test(csv_file1, csv_file2, output_file, comparisons_csv_file,
                     create_plots=False, plots_folder=None, 
                     group1_name='Grupo 1', group2_name='Grupo 2',
                     plot_dpi=300, plot_format='png'):
    """
    Realiza el test de Mann-Whitney U entre dos grupos
    
    Parámetros:
    csv_file1 (str): Ruta del archivo CSV del primer grupo
    csv_file2 (str): Ruta del archivo CSV del segundo grupo
    output_file (str): Ruta del archivo de texto para los resultados
    comparisons_csv_file (str): Ruta del archivo CSV para las comparaciones
    create_plots (bool): Si True, crea gráficos comparativos
    plots_folder (str): Carpeta donde guardar los gráficos
    group1_name (str): Nombre del grupo 1 para las leyendas
    group2_name (str): Nombre del grupo 2 para las leyendas
    plot_dpi (int): Resolución de las imágenes (300 recomendado para publicación)
    plot_format (str): Formato de imagen ('png', 'pdf', 'svg', 'tiff')
    """
    
    # Leer los archivos CSV
    try:
        data1 = pd.read_csv(csv_file1, encoding='latin1')
        data2 = pd.read_csv(csv_file2, encoding='latin1')
    except:
        try:
            data1 = pd.read_csv(csv_file1, encoding='iso-8859-1')
            data2 = pd.read_csv(csv_file2, encoding='iso-8859-1')
        except:
            import chardet
            with open(csv_file1, 'rb') as file:
                raw_data = file.read()
            result = chardet.detect(raw_data)
            data1 = pd.read_csv(csv_file1, encoding=result['encoding'])
            
            with open(csv_file2, 'rb') as file:
                raw_data = file.read()
            result = chardet.detect(raw_data)
            data2 = pd.read_csv(csv_file2, encoding=result['encoding'])

    # Obtener las variables (excluyendo 'evaluacion' y 'paciente')
    variables = data1.columns[2:]
    
    # Obtener evaluaciones únicas
    evaluaciones = sorted(data1['evaluacion'].unique())
    
    # Crear archivo para guardar resultados
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('RESULTADOS DEL TEST DE MANN-WHITNEY\n\n')
    
    # Crear lista para almacenar todos los resultados
    all_results = []
    
    # Realizar test para cada variable y cada evaluación
    for var in variables:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f'\n=== Variable: {var} ===\n')
        
        for eval_num in evaluaciones:
            try:
                # Obtener datos para esta evaluación
                grupo1 = data1[data1['evaluacion'] == eval_num][var].dropna()
                grupo2 = data2[data2['evaluacion'] == eval_num][var].dropna()
                
                # Verificar que hay suficientes datos
                if len(grupo1) < 2 or len(grupo2) < 2:
                    print(f"Advertencia: Datos insuficientes para {var} en evaluación {eval_num}")
                    continue
                
                print('Grupo1')
                print(grupo1)
                print('Grupo2')
                print(grupo2)
                
                # Realizar test de Mann-Whitney
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    statistic, p_value = stats.mannwhitneyu(
                        grupo1, 
                        grupo2,
                        alternative='two-sided'
                    )
                
                # Calcular estadísticos descriptivos
                median1 = grupo1.median()
                median2 = grupo2.median()
                iqr1_25 = grupo1.quantile(0.25)
                iqr1_75 = grupo1.quantile(0.75)
                iqr2_25 = grupo2.quantile(0.25)
                iqr2_75 = grupo2.quantile(0.75)
                iqr1 = grupo1.quantile(0.75) - grupo1.quantile(0.25)
                iqr2 = grupo2.quantile(0.75) - grupo2.quantile(0.25)
                
                # Guardar resultados
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f'\nEvaluación {eval_num}:\n')
                    f.write(f'Grupo 1 - Mediana (RIC): {median1:.2f} ({iqr1:.2f})\n')
                    f.write(f'Grupo 2 - Mediana (RIC): {median2:.2f} ({iqr2:.2f})\n')
                    f.write(f'Estadístico U: {statistic:.4f}\n')
                    f.write(f'p-valor: {p_value:.4f}\n')
                
                # Añadir a la lista de resultados
                all_results.append({
                    'Variable': var,
                    'Evaluacion': eval_num,
                    'Mediana_Grupo1': median1,
                    'RIC_Grupo1': iqr1,
                    'Q25_Grupo1': iqr1_25,
                    'Q75_Grupo1': iqr1_75,                    
                    'Mediana_Grupo2': median2,
                    'RIC_Grupo2': iqr2,
                    'Q25_Grupo2': iqr2_25,
                    'Q75_Grupo2': iqr2_75,
                    'Estadistico_U': statistic,
                    'p_valor': p_value
                })
                
            except Exception as e:
                print(f"Error procesando {var} en evaluación {eval_num}: {str(e)}")
    
    # Convertir resultados a DataFrame y guardar en CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(comparisons_csv_file, index=False, encoding='utf-8')
    else:
        print("No se generaron resultados válidos")
    
    # Crear gráficos si se solicita
    if create_plots:
        if plots_folder is None:
            plots_folder = str(Path(output_file).parent / 'graficos')
        
        print(f"\nGenerando gráficos en: {plots_folder}")
        create_comparison_plots(
            data1, data2, variables, evaluaciones, plots_folder,
            group1_name, group2_name, plot_dpi, plot_format
        )
        print("Gráficos generados exitosamente")

# Ejemplo de uso
if __name__ == "__main__":
    mann_whitney_test(
        csv_file1='C:/DATOS_CLINICOS/CH_FF.csv',
        csv_file2='C:/DATOS_CLINICOS/CH_R.csv',
        output_file='C:/DATOS_CLINICOS/resultados_mann_whitney_CH.txt',
        comparisons_csv_file='C:/DATOS_CLINICOS/comparaciones_mann_whitney_CH.csv',
        # Activar generación de gráficos
        create_plots=True,
        plots_folder='D:/90_CTTN/ArchivosPaper2_CH_CoP/graficos',
        group1_name='CH_FF',
        group2_name='CH_R',
        plot_dpi=300,  # Alta resolución para publicación
        plot_format='png'  # Opciones: 'png', 'pdf', 'svg', 'tiff'
    )