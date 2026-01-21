# 🎯 Challenge: Esencia del Cliente - Análisis y Segmentación con Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-yellow.svg)](https://colab.research.google.com/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/ramjasmaurya/medias-cost-prediction-in-foodmart)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">

![Logo Alura](https://www.aluracursos.com/assets/img/challenges/logos/challenges-logo-data.1712144089.svg)

**Segundo proyecto del Bootcamp de Data Science - Alura Latam**

[📊 Dataset](#-dataset) •
[🚀 Inicio Rápido](#-inicio-rápido) •
[📈 Metodología](#-metodología) •
[🔍 Resultados](#-resultados) •
[👨‍💻 Autor](#-autor)

</div>

---

## 📋 Descripción del Proyecto

Este proyecto se enfoca en **comprender profundamente el comportamiento de los clientes** mediante técnicas avanzadas de análisis de datos y machine learning. Utilizando algoritmos de clustering (agrupamiento), identificamos patrones significativos que permiten segmentar clientes en grupos homogéneos, facilitando estrategias de marketing personalizadas y mejorando la experiencia del cliente.

### 🎯 Objetivos Principales

- 🔍 **Análisis Exploratorio Profundo**: Visualizar y entender las características clave de los clientes
- 🤖 **Segmentación Inteligente**: Aplicar K-Means para agrupar clientes con comportamientos similares
- 📊 **Reducción de Dimensionalidad**: Implementar PCA para optimizar el análisis
- ✅ **Validación Rigurosa**: Evaluar la calidad de los clusters con múltiples métricas
- 💡 **Insights Accionables**: Generar recomendaciones estratégicas basadas en datos

### 💼 Impacto del Proyecto

La importancia de este análisis radica en su capacidad para **transformar datos en información accionable**. Al identificar y comprender diferentes segmentos de clientes, las empresas pueden:

- ✅ Desarrollar estrategias de marketing más efectivas y personalizadas
- ✅ Optimizar la asignación de recursos y presupuestos publicitarios
- ✅ Mejorar la retención y satisfacción del cliente
- ✅ Incrementar las ventas mediante ofertas dirigidas
- ✅ Tomar decisiones basadas en evidencia cuantitativa

---

## 📊 Dataset

### Fuente de Datos

Los datos fueron extraídos del conjunto de datos **"Media's Cost Prediction in Foodmart"** disponible en Kaggle:

🔗 **[Dataset en Kaggle](https://www.kaggle.com/datasets/ramjasmaurya/medias-cost-prediction-in-foodmart)**

### Características del Dataset

| Característica | Detalle |
|----------------|---------|
| **Origen** | Foodmart - Cadena de supermercados |
| **Tipo de datos** | Costos de medios, ventas y demografía de clientes |
| **Variables** | Categóricas y numéricas (mixtas) |
| **Idioma original** | Inglés (traducido al español) |
| **Formato** | CSV |

### Variables Principales Analizadas

```
📌 Variables Demográficas:
   • Escolaridad
   • Ocupación
   • Género
   • Estado Civil
   • Número de Hijos

📌 Variables Económicas:
   • Ingresos Anuales
   • Tipo de Miembro
   • Categoría de Alimentos
   • Tipo de Producto
```

---

## 🛠️ Tecnologías y Herramientas

### Stack Tecnológico

```python
# Análisis de Datos
pandas >= 1.3.0
numpy >= 1.19.0

# Visualización
matplotlib >= 3.3.0
seaborn >= 0.11.0

# Machine Learning
scikit-learn >= 1.0.0

# Ambiente de Desarrollo
Google Colab
Google Drive (almacenamiento)
```

### Técnicas de Machine Learning Implementadas

| Técnica | Propósito |
|---------|-----------|
| **K-Means** | Algoritmo de clustering no supervisado |
| **PCA** | Reducción de dimensionalidad |
| **StandardScaler** | Normalización de datos |
| **One-Hot Encoding** | Codificación de variables categóricas |

### Métricas de Validación

- 🎯 **Silhouette Score** (objetivo: ≥ 0.50)
- 📉 **Davies-Bouldin Index** (objetivo: ≤ 0.75)
- 📈 **Calinski-Harabasz Index** (maximizar)

---

## 🚀 Inicio Rápido

### Requisitos Previos

- ✅ Cuenta de Gmail (para acceder a Google Colab)
- ✅ Acceso a Google Drive
- ✅ Descarga del dataset desde Kaggle

### Instalación Paso a Paso

#### 1️⃣ Configurar Google Colab

```bash
# 1. Accede a Google Colab
https://colab.research.google.com/

# 2. Crea un nuevo notebook
Archivo → Nuevo Notebook

# 3. Renombra el notebook
"La esencia del cliente 1" (o el nombre de tu preferencia)
```

#### 2️⃣ Conectar con Google Drive

```python
# Montar Google Drive en Colab
from google.colab import drive
drive.mount('/content/drive')
```

#### 3️⃣ Descargar y Preparar el Dataset

1. **Descarga** el dataset desde [Kaggle](https://www.kaggle.com/datasets/ramjasmaurya/medias-cost-prediction-in-foodmart)
2. **Crea** un directorio en Google Drive: `Mi unidad/Datasets/Challenge-Cliente/`
3. **Sube** el archivo CSV al directorio creado

#### 4️⃣ Cargar Dependencias

```python
# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Configurar visualizaciones
%matplotlib inline
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
```

---

## 📈 Metodología

### Pipeline del Proyecto

```
📥 Carga de Datos
    ↓
🌐 Traducción al Español
    ↓
🔍 Exploración Visual (EDA)
    ↓
🔧 Preprocesamiento
    ↓
📊 Feature Engineering
    ↓
🤖 Clustering (K-Means)
    ↓
✅ Validación de Clusters
    ↓
📝 Análisis e Insights
```

---

### Fase 1: Carga y Traducción de Datos

#### Cargar Dataset desde Google Drive

```python
# Ruta al dataset en Google Drive
ruta_dataset = '/content/drive/MyDrive/Datasets/Challenge-Cliente/data.csv'

# Cargar datos
datos_raw = pd.read_csv(ruta_dataset)

# Vista preliminar
print(datos_raw.head())
print(f"Dimensiones: {datos_raw.shape}")
```

#### Traducción al Español

```python
# Diccionario de traducción (ejemplo)
traduccion_columnas = {
    'Education': 'Escolaridad',
    'Occupation': 'Ocupacion',
    'Member': 'Miembro',
    'Gender': 'Genero',
    'Marital_Status': 'Estado_Civil',
    'Num_Children': 'Num_Hijos',
    'Annual_Income': 'Ingresos_Anuales',
    'Food_Category': 'Categoria_Alimentos',
    'Type': 'Tipo'
}

# Aplicar traducción
datos_raw.rename(columns=traduccion_columnas, inplace=True)

# Exportar versión traducida
datos_raw.to_csv('/content/drive/MyDrive/Datasets/Challenge-Cliente/datos_traducidos.csv', index=False)
```

---

### Fase 2: Exploración Visual de Datos (EDA)

#### Análisis Estadístico Descriptivo

```python
# Estadísticas descriptivas
print(datos_raw.describe())

# Información general
print(datos_raw.info())

# Valores nulos
print(datos_raw.isnull().sum())
```

#### Visualizaciones Clave

**1. Distribución de Variables Numéricas**

```python
# Histograma de Ingresos Anuales
plt.figure(figsize=(10, 6))
sns.histplot(datos_raw['Ingresos_Anuales'], kde=True, bins=30)
plt.title('Distribución de Ingresos Anuales de Clientes')
plt.xlabel('Ingresos Anuales ($)')
plt.ylabel('Frecuencia')
plt.show()
```

**2. Análisis de Variables Categóricas**

```python
# Distribución por Género
plt.figure(figsize=(8, 5))
sns.countplot(data=datos_raw, x='Genero', palette='viridis')
plt.title('Distribución de Clientes por Género')
plt.show()
```

**3. Correlación entre Variables**

```python
# Matriz de correlación (solo variables numéricas)
plt.figure(figsize=(12, 8))
sns.heatmap(datos_raw.select_dtypes(include=[np.number]).corr(), 
            annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación')
plt.show()
```

**📝 Ejemplo de Observaciones:**

> *"Los clientes con mayor escolaridad tienden a tener ingresos anuales más altos. Existe una correlación positiva entre el número de hijos y el gasto en categoría de alimentos."*

---

### Fase 3: Preprocesamiento y Feature Engineering

#### Codificación de Variables Categóricas

**Opción 1: One-Hot Encoding**

```python
# Variables categóricas a codificar
categoricas = ['Escolaridad', 'Ocupacion', 'Genero', 'Estado_Civil']

# Aplicar One-Hot Encoding
datos_encoded = pd.get_dummies(datos_raw, columns=categoricas, drop_first=True)
```

**Opción 2: Label Encoding (Ordinal)**

```python
# Ejemplo: Escolaridad con orden jerárquico
escolaridad_map = {
    'Primaria': 1,
    'Secundaria': 2,
    'Universidad': 3,
    'Posgrado': 4
}

datos_raw['Escolaridad_Num'] = datos_raw['Escolaridad'].map(escolaridad_map)
```

#### Selección de Features Relevantes

```python
# Seleccionar entre 6 y 12 atributos más relevantes
features_seleccionadas = [
    'Escolaridad_Num',
    'Ingresos_Anuales',
    'Num_Hijos',
    'Edad',
    'Gasto_Total',
    'Frecuencia_Compra',
    'Categoria_Alimentos_Num',
    'Genero_M'  # si se usó One-Hot
]

X = datos_raw[features_seleccionadas]
```

#### Estandarización de Datos

```python
# Instanciar StandardScaler
scaler = StandardScaler()

# Ajustar y transformar
X_std = scaler.fit_transform(X)

print(f"Forma de X_std: {X_std.shape}")
# Output: (n_muestras, n_features)
```

---

### Fase 4: Clustering con K-Means

#### Determinación del Número Óptimo de Clusters

**Método del Codo (Elbow Method)**

```python
# Calcular inercia para diferentes números de clusters
inercias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_std)
    inercias.append(kmeans.inertia_)

# Gráfico del codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inercias, marker='o', linewidth=2)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para Determinar k Óptimo')
plt.grid(True)
plt.show()
```

#### Validación con Múltiples Métricas

```python
# Evaluar de 3 a 10 clusters
resultados = []

for k in range(3, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_std)
    
    # Calcular métricas
    silhouette = silhouette_score(X_std, labels)
    davies_bouldin = davies_bouldin_score(X_std, labels)
    calinski = calinski_harabasz_score(X_std, labels)
    
    resultados.append({
        'k': k,
        'Silhouette': silhouette,
        'Davies-Bouldin': davies_bouldin,
        'Calinski-Harabasz': calinski
    })

# Crear DataFrame con resultados
df_metricas = pd.DataFrame(resultados)
print(df_metricas)
```

**Criterios de Selección:**

- ✅ **Silhouette** ≥ 0.50 (mayor es mejor)
- ✅ **Davies-Bouldin** ≤ 0.75 (menor es mejor)
- ✅ **Calinski-Harabasz**: maximizar

---

### Fase 5: Validación de Estructura y Estabilidad

#### 1️⃣ Validación de Estructura (Baseline Aleatorio)

```python
# Generar datos aleatorios con la misma forma que X_std
random_data = np.random.rand(*X_std.shape)

# Aplicar KMeans al baseline
k_optimo = 4  # ejemplo: mejor k encontrado
kmeans_random = KMeans(n_clusters=k_optimo, random_state=42)
labels_random = kmeans_random.fit_predict(random_data)

# Calcular métricas en baseline
sil_random = silhouette_score(random_data, labels_random)
db_random = davies_bouldin_score(random_data, labels_random)
ch_random = calinski_harabasz_score(random_data, labels_random)

print(f"Baseline Aleatorio - Silhouette: {sil_random:.3f}")
print(f"Datos Reales - Silhouette: {silhouette_score(X_std, labels):.3f}")
# Asegurar que X_std >> random_data
```

#### 2️⃣ Validación de Estabilidad (Cross-Validation)

```python
# Dividir X_std en 5 partes iguales
splits = np.array_split(X_std, 5)

metricas_estabilidad = []

for i, split in enumerate(splits):
    kmeans_split = KMeans(n_clusters=k_optimo, random_state=42)
    labels_split = kmeans_split.fit_predict(split)
    
    sil = silhouette_score(split, labels_split)
    db = davies_bouldin_score(split, labels_split)
    ch = calinski_harabasz_score(split, labels_split)
    
    metricas_estabilidad.append({
        'Split': i+1,
        'Silhouette': sil,
        'Davies-Bouldin': db,
        'Calinski-Harabasz': ch
    })

df_estabilidad = pd.DataFrame(metricas_estabilidad)

# Calcular variación porcentual
variacion_sil = df_estabilidad['Silhouette'].std() / df_estabilidad['Silhouette'].mean() * 100
print(f"Variación en Silhouette: {variacion_sil:.2f}%")
# Objetivo: variación < 5%
```

---

### Fase 6: Asignación de Clusters al Dataset

```python
# Instanciar modelo final con k óptimo
kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)

# Ajustar y predecir
datos_raw['cluster'] = kmeans_final.fit_predict(X_std)

# Verificar distribución de clusters
print(datos_raw['cluster'].value_counts().sort_index())
```

---

### Fase 7: Análisis e Interpretación de Clusters

#### Visualización de Clusters

**1. Gráfico de Dispersión 2D**

```python
# Seleccionar dos variables clave
plt.figure(figsize=(12, 7))
sns.scatterplot(data=datos_raw, 
                x='Ingresos_Anuales', 
                y='Gasto_Total',
                hue='cluster', 
                palette='Set2', 
                s=100, 
                alpha=0.7)
plt.title('Segmentación de Clientes: Ingresos vs Gasto Total')
plt.xlabel('Ingresos Anuales ($)')
plt.ylabel('Gasto Total ($)')
plt.legend(title='Cluster')
plt.show()
```

**2. Visualización 3D (con PCA)**

```python
from mpl_toolkits.mplot3d import Axes3D

# Reducir a 3 componentes principales
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_std)

# Gráfico 3D
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_pca[:, 0], 
                     X_pca[:, 1], 
                     X_pca[:, 2],
                     c=datos_raw['cluster'], 
                     cmap='viridis', 
                     s=50, 
                     alpha=0.6)

ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.set_title('Clusters en Espacio PCA 3D')
plt.colorbar(scatter, label='Cluster')
plt.show()
```

#### Perfiles de Clusters

```python
# Análisis estadístico por cluster
perfiles = datos_raw.groupby('cluster')[features_seleccionadas].mean()
print(perfiles)

# Visualizar perfiles
perfiles.T.plot(kind='bar', figsize=(14, 8), colormap='tab10')
plt.title('Perfil Promedio de Cada Cluster')
plt.ylabel('Valor Promedio Estandarizado')
plt.xlabel('Features')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

---

## 🔍 Resultados

### Ejemplo de Descripción de Clusters

#### 📊 **Cluster 0: Clientes Premium**
- **Características**:
  - Ingresos anuales superiores a $80,000
  - Alta escolaridad (Universidad/Posgrado)
  - Gasto elevado en productos premium
  - Edad promedio: 40-55 años
  
- **Estrategia sugerida**:
  - Programa de lealtad exclusivo
  - Comunicación personalizada de productos premium
  - Eventos VIP

#### 📊 **Cluster 1: Familias Jóvenes**
- **Características**:
  - Ingresos medios ($40,000 - $60,000)
  - Número de hijos: 2-3
  - Mayor gasto en categoría de alimentos
  - Edad promedio: 30-40 años
  
- **Estrategia sugerida**:
  - Promociones familiares
  - Descuentos en productos infantiles
  - Programas de ahorro

#### 📊 **Cluster 2: Compradores Ocasionales**
- **Características**:
  - Ingresos bajos-medios (< $40,000)
  - Frecuencia de compra baja
  - Sensibilidad al precio
  - Sin hijos o 1 hijo
  
- **Estrategia sugerida**:
  - Cupones y descuentos
  - Comunicación de ofertas especiales
  - Programa de puntos

#### 📊 **Cluster 3: Seniors Estables**
- **Características**:
  - Ingresos medios-altos por jubilación
  - Edad > 60 años
  - Compras regulares pero moderadas
  - Prefieren productos de calidad
  
- **Estrategia sugerida**:
  - Servicio personalizado
  - Productos de salud y bienestar
  - Facilidades de entrega a domicilio

---

## 📚 Recursos y Referencias

### Documentación Oficial

- [Scikit-learn - K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Scikit-learn - PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Tutoriales Recomendados

- [K-Means Clustering - Antonio Richaud](https://antonio-richaud.com/blog/archivo/publicaciones/12-k-means.html)
- [PCA (Análisis de Componentes Principales) - Antonio Richaud](https://antonio-richaud.com/blog/archivo/publicaciones/29-pca.html)
- [Google Colab - Guía Oficial](https://colab.research.google.com/notebooks/intro.ipynb)

### Papers y Artículos

- **K-Means Clustering**: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- **PCA**: Pearson, K. (1901). "On lines and planes of closest fit to systems of points in space"

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto:

1. **Fork** el repositorio
2. Crea tu **feature branch** (`git checkout -b feature/MejoraMagica`)
3. **Commit** tus cambios (`git commit -m 'Añade nueva métrica de validación'`)
4. **Push** a la rama (`git push origin feature/MejoraMagica`)
5. Abre un **Pull Request**

### Ideas de Contribución

- 📊 Implementar otros algoritmos de clustering (DBSCAN, Hierarchical)
- 🎨 Mejorar visualizaciones con Plotly (interactividad)
- 📈 Añadir análisis de series temporales
- 🧪 Integrar pruebas unitarias
- 📝 Traducir documentación a otros idiomas

---

## 👨‍💻 Autor

<div align="center">

**Darwin Manuel Ovalles Cesar**

<p align="center">
<a href="https://www.linkedin.com/in/darwin-manuel-ovalles-cesar-dev" target="_blank">
<img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="LinkedIn - Darwin Ovalles" height="40" width="50" />
</a>
</p>

💼 **LinkedIn**: [darwin-manuel-ovalles-cesar-dev](https://www.linkedin.com/in/darwin-manuel-ovalles-cesar-dev)  
🌐 **GitHub**: [@dovalless](https://github.com/dovalless)  
📧 **Email**: Disponible en LinkedIn

---

*"Este proyecto es una contribución con todo el amor del mundo para aquellos que buscan formarse en el fascinante ámbito de la Ciencia de Datos. Espero que mi trabajo pueda servir como una guía y recurso valioso para cualquier persona interesada en mejorar sus habilidades y conocimientos en esta área."*

**#aluraChallengeEsenciaDelCliente**

</div>

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

## 🏆 Insignia del Challenge

<div align="center">

![Insignia Challenge](./imagenes/medallita.png)

**Bootcamp de Data Science - Alura Latam**

</div>

---

## 🙏 Agradecimientos

- **Alura Latam** - Por el excelente programa de formación en Data Science
- **Kaggle** - Por proporcionar datasets de calidad para practicar
- **Comunidad Open Source** - Por las herramientas y librerías utilizadas
- **Antonio Richaud** - Por los excelentes tutoriales de K-Means y PCA

---

<div align="center">

**⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub ⭐**

**🚀 ¡Feliz análisis de datos! 🚀**

---

Desarrollado con 💚 y ☕ por [Darwin Ovalles](https://www.linkedin.com/in/darwin-manuel-ovalles-cesar-dev)

</div>
