# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 16:28:22 2025

@author: HP
"""

import numpy as np

def balas_hammer_vogel(offre, demande, couts):
    """
    Implémente la Méthode d'Approximation de Vogel (VAM) / Balas-Hammer
    pour trouver une solution de base initiale à un problème de transport.

    """
    # Convertir les entrées en numpy arrays pour faciliter les opérations
    offre_restante = np.array(offre, dtype=float)
    demande_restante = np.array(demande, dtype=float)
    couts_matrice = np.array(couts, dtype=float)

    n_offre = len(offre)
    n_demande = len(demande)

    # Initialiser la matrice d'allocation
    allocation_matrice = np.zeros((n_offre, n_demande), dtype=float)

    # Suivre les lignes et colonnes satisfaites
    lignes_satisfaites = np.zeros(n_offre, dtype=bool)
    colonnes_satisfaites = np.zeros(n_demande, dtype=bool)

    print("--- Début de la Méthode VAM (Balas-Hammer) ---")

    # La boucle continue tant qu'il reste de l'offre ou de la demande non satisfaite
    while not all(lignes_satisfaites) or not all(colonnes_satisfaites):

        # --- Étape 1 : Calculer les Pénalités (Différences) ---

        penalites_lignes = []
        for i in range(n_offre):
            if not lignes_satisfaites[i]:
                # Coûts non barrés de la ligne i
                couts_ligne = couts_matrice[i, ~colonnes_satisfaites]
                
                # Nécessite au moins deux coûts pour calculer la différence
                if len(couts_ligne) >= 2:
                    # Trier et prendre les deux plus petits
                    sorted_costs = np.sort(couts_ligne)
                    penalite = sorted_costs[1] - sorted_costs[0]
                else:
                    penalite = -1  # Marqueur pour les lignes à un seul élément restant
                penalites_lignes.append(penalite)
            else:
                penalites_lignes.append(-np.inf) # Ligne déjà satisfaite

        penalites_colonnes = []
        for j in range(n_demande):
            if not colonnes_satisfaites[j]:
                # Coûts non barrés de la colonne j
                couts_colonne = couts_matrice[~lignes_satisfaites, j]
                
                if len(couts_colonne) >= 2:
                    sorted_costs = np.sort(couts_colonne)
                    penalite = sorted_costs[1] - sorted_costs[0]
                else:
                    penalite = -1
                penalites_colonnes.append(penalite)
            else:
                penalites_colonnes.append(-np.inf) # Colonne déjà satisfaite

        # --- Étape 2 : Trouver la Pénalité Maximale ---

        max_penalite_ligne = max(penalites_lignes)
        max_penalite_colonne = max(penalites_colonnes)
        
        penalite_max = max(max_penalite_ligne, max_penalite_colonne)
        
        # Si la pénalité maximale est -1 ou -inf, cela signifie qu'il ne reste
        # qu'une seule ligne/colonne non satisfaite. On arrête la boucle si plus rien n'est faisable.
        if penalite_max <= 0 and not (np.any(~lignes_satisfaites) or np.any(~colonnes_satisfaites)):
             break # Toutes les lignes/colonnes sont satisfaites, ou il ne reste que des cas triviaux

        # --- Étape 3 : Choisir la Cellule (i, j) pour l'Allocation ---

        ligne_choisie = -1
        colonne_choisie = -1
        
        # Le maximum se trouve dans une ligne
        if max_penalite_ligne >= max_penalite_colonne:
            ligne_choisie = penalites_lignes.index(max_penalite_ligne)
            i = ligne_choisie
            
            # Dans cette ligne, trouver le coût minimum parmi les colonnes NON satisfaites
            couts_valides = couts_matrice[i, :]
            
            # Filtrer pour ne considérer que les colonnes non barrées
            min_cout = np.inf
            for j_temp in range(n_demande):
                if not colonnes_satisfaites[j_temp] and couts_valides[j_temp] < min_cout:
                    min_cout = couts_valides[j_temp]
                    colonne_choisie = j_temp
            j = colonne_choisie
            
        # Le maximum se trouve dans une colonne
        else: # max_penalite_colonne > max_penalite_ligne
            colonne_choisie = penalites_colonnes.index(max_penalite_colonne)
            j = colonne_choisie
            
            # Dans cette colonne, trouver le coût minimum parmi les lignes NON satisfaites
            couts_valides = couts_matrice[:, j]
            
            min_cout = np.inf
            for i_temp in range(n_offre):
                if not lignes_satisfaites[i_temp] and couts_valides[i_temp] < min_cout:
                    min_cout = couts_valides[i_temp]
                    ligne_choisie = i_temp
            i = ligne_choisie

        # Si pour une raison ou une autre (comme un tableau vide), i ou j est -1, on sort
        if i == -1 or j == -1:
            break
            
        # --- Étape 4 : Allocation et Mise à Jour ---

        allocation = min(offre_restante[i], demande_restante[j])

        allocation_matrice[i, j] = allocation
        offre_restante[i] -= allocation
        demande_restante[j] -= allocation

        print(f"Allouer {allocation} à la cellule ({i}, {j}) avec un coût de {couts_matrice[i, j]}")
        print(f"Pénalité maximale choisie : {penalite_max}")

        # --- Étape 5 : Barrer la Ligne ou la Colonne ---
        if offre_restante[i] == 0:
            lignes_satisfaites[i] = True
            print(f"Ligne {i} (Offre) satisfaite/barrée.")
        
        # NOTE : Si les deux sont nuls, on barre SEULEMENT l'un des deux (pour éviter la dégénérescence)
        # Ici, on laisse la ligne i barrée et la colonne j non barrée.
        elif demande_restante[j] == 0:
            colonnes_satisfaites[j] = True
            print(f"Colonne {j} (Demande) satisfaite/barrée.")
            
    print("--- Fin de la Méthode VAM ---")
    return allocation_matrice.tolist()

# --- Exemple d'utilisation ---

# Offres des sources (S1, S2, S3)
offre_exemple = [20, 17,13]
# Demandes des destinations (D1, D2, D3, D4)
demande_exemple = [12,10, 15,13]

couts_exemple = [
    [3, 6, 4,8],
    [3, 4, 7, 9],
    [9, 4, 5, 6]
]

resultat_allocation = balas_hammer_vogel(offre_exemple, demande_exemple, couts_exemple)

## Affichage du résultat
print("\nMatrice d'Allocation Finale:")
for row in resultat_allocation:
    # Convertir les floats en entiers pour l'affichage (si l'allocation est entière)
    print([int(x) for x in row])

# Calcul du coût total de transport (Solution Initiale)
cout_total = 0
for i in range(len(offre_exemple)):
    for j in range(len(demande_exemple)):
        cout_total += resultat_allocation[i][j] * couts_exemple[i][j]

print(f"\nCoût total de transport de la solution VAM : {int(cout_total)}")