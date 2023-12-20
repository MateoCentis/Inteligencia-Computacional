import seaborn as sns
import matplotlib.pyplot as plt

# Datos de ejemplo
nombres = ['Alice', 'Bob', 'Charlie']
puntuaciones = [95, 87, 92]

# Crear un gráfico de barras con seaborn
sns.barplot(x=nombres, y=puntuaciones)
plt.xlabel('Nombre')
plt.ylabel('Puntuación')
plt.title('Puntuaciones de los participantes')
plt.show()