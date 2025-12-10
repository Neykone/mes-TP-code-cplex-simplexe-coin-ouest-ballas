# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:40:14 2025

@author: HP
"""

def coin_nord_ouest(offre, demande):
    """
    Implémente la méthode du Coin Nord-Ouest pour trouver une solution de base
    initiale à un problème de transport.

    :param offre: Liste des quantités d'offre disponibles (offre[i]).
    :param demande: Liste des quantités de demande requises (demande[j]).
    :return: Une matrice (liste de listes) représentant les allocations.
    """
    # Crée des copies pour ne pas modifier les listes originales
    offre_restante = list(offre)
    demande_restante = list(demande)

    n_offre = len(offre_restante)
    n_demande = len(demande_restante)

    # Initialise la matrice d'allocation (taille n_offre x n_demande) avec des zéros
    allocation_matrice = [[0] * n_demande for _ in range(n_offre)]

   
    i = 0  
    j=0

    print("--- Étapes d'Allocation ---")

    # Tant qu'il reste de l'offre à considérer et de la demande à satisfaire
    while i < n_offre and j < n_demande:
        # 1. Déterminer la quantité d'allocation
        allocation = min(offre_restante[i], demande_restante[j])

        # 2. Affecter l'allocation
        allocation_matrice[i][j] = allocation
        print(f"Allouer {allocation} à la cellule ({i}, {j})")

        # 3. Mettre à jour les quantités restantes
        offre_restante[i] -= allocation
        demande_restante[j] -= allocation

        # 4. Avancer : épuisement de l'offre ou de la demande
        if offre_restante[i] == 0:
            # L'offre de la source i est épuisée, passer à la source suivante
            print(f"Source {i} épuisée.")
            i += 1
        elif demande_restante[j] == 0:
            # La demande de la destination j est satisfaite, passer à la destination suivante
            print(f"Destination {j} satisfaite.")
            j += 1
        
        # Note : Si les deux sont à 0 (cas dégénéré), on avance dans l'offre (i += 1)
        # pour satisfaire la condition du pseudocode : Si offre[i] == 0 -> i = i + 1.

    print("--- Fin de la Méthode ---")
    return allocation_matrice

# --- Exemple d'utilisation ---

"""# Offres des 4 fournisseurs/sources (o1, o2, o3,o4)
offre = [12,11,14,8]
# Demandes des 5 destinations/magasins (D1, D2, D3, D4,D)
demande = [10, 11, 15, 5,4]"""


nboffre = int(input("entrer le nombre d'usines :"))
offre=[]
for i in range(0,nboffre) :
    off=int(input("entrer un nombre :"))
    offre.append(off)
    
nbdemande=int(input("entrer le nombre de demande :"))    

demande=[]

for j in range(0,nbdemande) :
    dem=int(input("entre une demande : "))
    demande.append(dem) 
   
    


resultat_allocation = coin_nord_ouest(offre, demande)

## Affichage du résultat
print("\nMatrice d'Allocation Finale:")
# Affichage formaté pour la lisibilité
for row in resultat_allocation:
    print(row)