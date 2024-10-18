import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Cargar los datos generados previamente
df = pd.read_csv('clientes_favoritos.csv')

# PREPARACIÓN DE DATOS
# Verificación de valores de 'Favorito' (1=favorito, 0=no favorito) y visualización de los primeros datos
print(df['Favorito'].value_counts())
print(df.head())

# Creamos una matriz item-usuario con los datos necesarios (favoritos binarios)
iu = df.pivot_table(index='Producto_ID', columns='Cliente_ID', values='Favorito', fill_value=0)

# Conservar información de imagen para cada Producto_ID
images = df[['Producto_ID', 'Imagen']].drop_duplicates().set_index('Producto_ID')

# CÁLCULO DE LA SIMILITUD DE ITEMS
similitud_items = cosine_similarity(iu.to_numpy())

# CREACIÓN DE UNA MATRIZ DE RECOMENDACIONES
matriz_recomendaciones = pd.DataFrame(similitud_items, index=iu.index, columns=iu.index)

# Convertir el DataFrame de una matriz a un formato largo
matriz_recomendaciones_long = matriz_recomendaciones.stack().rename_axis(['id1', 'id2']).reset_index(name='similitud')
matriz_recomendaciones_long = matriz_recomendaciones_long[matriz_recomendaciones_long['id1'] != matriz_recomendaciones_long['id2']]
matriz_recomendaciones_long = matriz_recomendaciones_long[matriz_recomendaciones_long['id1'] < matriz_recomendaciones_long['id2']]

# Unir las imágenes de 'Producto_ID_2'
matriz_recomendaciones_long = matriz_recomendaciones_long.join(images, on='id2')

# Guardar la matriz de recomendación a disco
matriz_recomendaciones_long.to_pickle("matriz_recomendaciones_long.pkl")

# Guardar las recomendaciones en un archivo JSON
matriz_recomendaciones_long.to_json("matriz_recomendaciones_long.json", orient="records")

output_recommendations

print(matriz_recomendaciones.head())

