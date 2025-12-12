# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import math
from scipy.optimize import linprog
import matplotlib.pyplot as plt




# --- La fonction corrigée (basée sur l'analyse de votre tentative) ---
def generate(A, b, c):
    """
    Construit le tableau initial du simplexe pour un problème de minimisation.
    Structure: [ A | I | b ]
               [ -c | 0 | 0 ]
    """
    nl, nc = A.shape
    I = np.eye(nl)
    # Correction: np.zeros a besoin d'un tuple (nl+1, nc+nl+1)
    T = np.zeros((nl + 1, nc + nl + 1)) 
    
    # Partie contraintes [A | I | b]
    T[:nl, :nc] = A          # Matrice A (variables de décision)
    T[:nl, nc:nc+nl] = I     # Matrice Identité I (variables d'écart)
    T[:nl, nc+nl] = b        # Vecteur b (second membre)
    
    # Partie Coûts réduits [ -c | 0...0 | 0 ]
    T[nl, :nc] = c          # Coûts réduits des variables de décision (c_j)
    T[nl, nc+nl] = 0         # Z initial = 0
    
    return T

# ----------------------------------------------------------------------

"""# Arguments pour l'appel:
A_input = np.array([[2, 8], [4, 6]])
b_input = np.array([1, 5])
c_input = np.array([4, 5])

# Appel corrigé
tableau_initial = generate(A_input, b_input, c_input)

print("--- Tableau Initial du Simplexe Généré ---")
print(tableau_initial)"""

def positivite(v):
    """
    Vérifie si toutes les composantes d'un vecteur sont positives ou nulles (>= 0).
    Sert à déterminer si la solution optimale est trouvée dans l'algorithme du simplexe.
    
    """
    
    # Vérifie si toutes les composantes sont positives ou nulles
    is_positive = np.all(v >= 0)
    
    # Trouve la valeur minimale
    min_value = np.min(v)
    
    # Trouve l'indice de la première occurrence de la valeur minimale
    min_index = np.argmin(v)
    
    return is_positive, min_value, min_index

vecteur = np.array([0, 5, -2, 10, 0, -4])

"""p,minvalue,minindex=positivite(vecteur)"""


def rapportmin(b, a):
    """
    Calcule et retourne l'indice de la ligne du rapport minimum positif.
    
    Implémente la règle du rapport minimum: r = argmin { b_i / a_i : a_i > 0 }

    Returns:
        int: Indice r de la ligne pivot.
        
    """
    
    # Étape 1: Créer un masque pour sélectionner uniquement les a_i strictement positifs.
    positive_mask = a > 0
    
    # Étape 2: Vérifier la condition d'arrêt (étape 3a de l'énoncé)
    # Si tous les a_i <= 0, le problème est non borné.
    if not np.any(positive_mask):
        raise ValueError("L'objectif n'est pas borné inférieurement (tous les éléments de la colonne pivot sont <= 0).")
        
    # Étape 3: Calculer les rapports (b_i / a_i) uniquement pour les éléments positifs.
    # On assure que les vecteurs sont plats avant l'opération de division.
    b_positive = b[positive_mask].flatten()
    a_positive = a[positive_mask].flatten()
    
    ratios = b_positive / a_positive
    
    # Étape 4: Trouver l'indice du rapport minimum dans ce sous-ensemble (ratios).
    min_ratio_index_in_subset = np.argmin(ratios)
    
    # Étape 5: Convertir l'indice trouvé dans le sous-ensemble en l'indice original (r)
    # dans le vecteur initial b et a.
    
    # np.where(positive_mask)[0] donne les indices ORIGINAUX qui ont passé le masque.
    original_indices = np.where(positive_mask)[0]
    
    # L'indice r est l'élément qui correspond au minimum dans le sous-ensemble trié.
    r = original_indices[min_ratio_index_in_subset]
    
    return r

"""# --- Exemples de Test ---

# 1. Cas Normal (Minimum Positif)
b1 = np.array([10, 20, 15])
a1 = np.array([2, 5, 3])
# Rapports : [10/2=5, 20/5=4, 15/3=5]
# Minimum : 4 à l'indice 1
r1 = rapportmin(b1, a1)
print(f"Test 1: b={b1}, a={a1} -> Indice min positif: {r1}") 
# Résultat attendu: 1

# 2. Cas avec des a_i non positifs
b2 = np.array([10, 20, 30, 40])
a2 = np.array([-2, 5, 10, -5])
# Rapports positifs : [20/5=4, 30/10=3]
# Minimum : 3 à l'indice original 2
r2 = rapportmin(b2, a2)
print(f"Test 2: b={b2}, a={a2} -> Indice min positif: {r2}")
# Résultat attendu: 2

# 3. Cas de non borné (tous les a_i sont négatifs ou nuls)
b3 = np.array([1, 1, 1])
a3 = np.array([-1, 0, -5])
try:
    rapportmin(b3, a3)
except ValueError as e:
    print(f"Test 3 (Non borné): {e}") """
    


def pivotgauss(T, r, s):
    """
    Effectue l'opération de pivot de Gauss sur le tableau simplexe T.
    
    Cette fonction met à jour le tableau T en utilisant la ligne r et la colonne s 
    pour transformer la colonne pivot en un vecteur unitaire, assurant que la nouvelle 
    solution de base reste réalisable.

    Returns:
        np.ndarray: Le nouveau tableau simplexe après pivotage.
    """
    
    # 1. Créer une copie du tableau pour éviter de modifier le tableau original en place
    T_new = T.copy()
    
    # Le pivot est l'élément à l'intersection de la ligne r et de la colonne s
    alpha_rs = T_new[r, s]
    
    if alpha_rs == 0:
        # Cette situation devrait être prévenue par la fonction rapportmin
        raise ValueError("Le pivot est zéro. Impossible de diviser.")
        
    # --- Étape 4(a) : Normalisation de la ligne pivot ---
    # Diviser la ligne pivot par le pivot alpha_rs
    T_new[r, :] = T_new[r, :] / alpha_rs
    
    # --- Étape 4(b) et 4(c) : Opérations sur les autres lignes ---
    # Mettre à zéro les autres éléments de la colonne pivot (incluant la ligne de coûts)
    
    m_plus_1, _ = T_new.shape
    
    # Parcourir toutes les lignes i, y compris la ligne de coûts
    for i in range(m_plus_1):
        if i != r:
            # Coefficient alpha_is à annuler
            alpha_is = T_new[i, s]
            
            # Nouvelle Ligne i = Ancienne Ligne i - alpha_is * Nouvelle Ligne Pivot r
            
            T_new[i, :] = T_new[i, :] - alpha_is * T_new[r, :]
            
    return T_new

"""

# Tableau Initial:
# [[ 2.  8.  1.  0.  1.]  <-- Ligne 0
#  [ 4.  6.  0.  1.  5.]  <-- Ligne 1
#  [-4. -5.  0.  0.  0.]] <-- Ligne 2 (coûts)
T_initial = np.array([
    [ 2.,  8.,  1.,  0.,  1.],
    [ 4.,  6.,  0.,  1.,  5.],
    [-4., -5.,  0.,  0.,  0.]
])


r_pivot = 0
s_pivot = 1

T_pivot = pivotgauss(T_initial, r_pivot, s_pivot)

print(f"--- Tableau Simplexe après Pivotage (r={r_pivot}, s={s_pivot}) ---")
print(T_pivot.round(3))    
"""



def simplexe_primal(A, b, c, max_iter=100):
    """
    Implémente l'algorithme du simplexe primal pour un problème de minimisation.

    Returns:
        tuple: (Valeur optimale de z, Vecteur de solution x optimal, Tableau final)
               Retourne (None, None, T) en cas d'échec (non borné ou max_iter atteint).
    """
    
    # 0. Récupération des dimensions
    m, n = A.shape # m contraintes, n variables de décision
    
    # 1. Initialisation : Construction du tableau initial
    try:
        T = generate(A, b, c)
    except Exception as e:
        print(f"Erreur lors de la construction du tableau initial: {e}")
        return None, None, None

    print("--- Démarrage de l'algorithme du Simplexe ---")

    for iteration in range(max_iter):
        print(f"\n✨ Itération {iteration + 1}")
        
        # Le vecteur de coûts réduits (dernière ligne, sans le 'z')
        costs = T[-1, :-1]
        
        # 2. Choix de la colonne pivot (variable à entrer en base)
        # (a) Si c_j >= 0, STOP : Solution optimale trouvée
        is_optimal, min_cost, s = positivite(costs)
        
        if is_optimal:
            print("✅ STOP : Solution optimale trouvée (tous les coûts réduits sont >= 0).")
            break
        
        # (b) Sinon, choisir la colonne s avec le coût réduit le plus négatif
        # L'indice s est l'indice de la colonne pivot
        print(f"   Variable entrante (colonne pivot s): {s + 1} (coût réduit: {min_cost:.4f})")
        
        # 3. Choix de la ligne pivot (variable à sortir de la base)
        # Colonne pivot (a) et vecteur b
        a_s = T[:-1, s] 
        b_col = T[:-1, -1]
        
        try:
            r = rapportmin(b_col, a_s)
            print(f"   Variable sortante (ligne pivot r): {r + 1}")
            
        # (a) Si a_is <= 0, STOP : Non borné inférieurement
        except ValueError as e:
            print(f"❌ STOP : {e}")
            return None, None, T
            
        # 4. Mise à jour du tableau
        T = pivotgauss(T, r, s)
        
    else:
        # Si la boucle a atteint max_iter sans break
        print(f"\n⚠️ STOP : Nombre maximum d'itérations ({max_iter}) atteint (non convergence ou boucle infinie).")
        return None, None, T
        
    # --- Extraction de la Solution Optimale (après convergence) ---
    
    # Valeur optimale de Z se trouve en T[-1, -1]
    z_optimal = T[-1, -1]
    
    # Vecteur solution x (n variables de décision + m variables d'écart)
    x_solution = np.zeros(n + m)
    
    # Identifier les variables de base (en trouvant les colonnes qui forment l'identité)
    for j in range(n + m): # Parcourt les colonnes de variables
        column = T[:-1, j] # Colonne des contraintes
        
        # Une colonne est de base si elle contient un seul 1 et des 0 ailleurs
        if (np.sum(column == 1) == 1) and (np.sum(column == 0) == m - 1):
            # Trouver la ligne où se trouve le 1
            basic_row_index = np.where(column == 1)[0][0]
            
            # La valeur de cette variable de base est dans la colonne b de cette ligne
            x_solution[j] = T[basic_row_index, -1]

    # Solution pour les variables de décision x_1, ..., x_n
    x_optimal = x_solution[:n]
    
    print("\n--- Résultat Final ---")
    print(f"Valeur optimale de z: {z_optimal:.4f}")
    print(f"Solution optimale (variables de décision x): {x_optimal}")
    
    return z_optimal, x_optimal, T


A = np.array([[1, 1], [1, 2]])
b = np.array([4, 6])
c = np.array([-3, -2])

# Exécution de l'algorithme
z_opt, x_opt, T_final = simplexe_primal(A, b, c)


res=linprog(c, A_ub=A, b_ub=b)
print(res)
print('\nValeur optimale (Min Z):', res.fun, '\nVariables X (x1, x2, ...):', res.x)





     
     