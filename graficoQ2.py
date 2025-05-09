import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Documenti
D1 = np.array([1, 1, 0])  # retrieval + model
D2 = np.array([1, 0, 1])  # retrieval + data
D3 = np.array([0, 1, 1])  # model + data

# Query "data model" con piccolo offset su asse x per distinguere da D3
epsilon = 0.05
Q = np.array([epsilon, 1, 1])

# Calcolo similarità coseno
docs = np.array([D1, D2, D3])
sims = cosine_similarity([Q], docs).flatten()

# Etichette e colori
labels = ['D1', 'D2', 'D3']
colors = ['blue', 'green', 'orange']

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Vettore query
ax.quiver(0, 0, 0, *Q, color='red', label='Q: data model', linewidth=2, arrow_length_ratio=0.1)
ax.text(Q[0], Q[1], Q[2], 'Q', color='red')

# Linee tratteggiate per la query (offset visivo)
qx, qy, qz = Q
ax.plot([qx, qx], [qy, qy], [0, qz], color='red', linestyle='dashed')
ax.plot([qx, qx], [0, qy], [qz, qz], color='red', linestyle='dashed')
ax.plot([0, qx], [qy, qy], [qz, qz], color='red', linestyle='dashed')

# Vettori documenti
for vec, label, color, sim in zip(docs, labels, colors, sims):
    ax.quiver(0, 0, 0, *vec, color=color, linewidth=2, arrow_length_ratio=0.1)
    ax.text(vec[0], vec[1], vec[2], f'{label}\nSim={sim:.2f}', color=color)
    
    x, y, z = vec
    ax.plot([x, x], [y, y], [0, z], color=color, linestyle='dashed')
    ax.plot([x, x], [0, y], [z, z], color=color, linestyle='dashed')
    ax.plot([0, x], [y, y], [z, z], color=color, linestyle='dashed')

# Setup assi
ax.set_xlim([0, 1.2])
ax.set_ylim([0, 1.2])
ax.set_zlim([0, 1.2])
ax.set_xlabel('retrieval')
ax.set_ylabel('model')
ax.set_zlabel('data')
ax.set_title('Funzione di reperimento (coseno) - Query: "data model"')

plt.tight_layout()
plt.show()
