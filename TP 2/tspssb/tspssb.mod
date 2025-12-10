/*********************************************
 * OPL 22.1.1.0 Model
 * Author: HP
 * Creation Date: 28 nov. 2025 at 10:47:33
 *********************************************/
 
int n = ...; 
range Villes = 1..n;
range Villes_sauf_Ville1 = 2..n;

float d[Villes][Villes] = ...;

dvar boolean x[Villes][Villes];

dvar boolean u[Villes_sauf_Ville1][Villes_sauf_Ville1];

minimize
    sum(i in Villes, j in Villes: i != j) d[i][j] * x[i][j];

subject to {
    
    forall(i in Villes)
        ctDepart:
            sum(j in Villes: i != j) x[i][j] == 1;
   
    forall(j in Villes)
        ctArrivee:
            sum(i in Villes: i != j) x[i][j] == 1;
  
    forall(i in Villes_sauf_Ville1, j in Villes_sauf_Ville1: i != j)
        ctLiaison1:
            u[i][j] >= x[i][j];
   
    forall(i in Villes_sauf_Ville1, j in Villes_sauf_Ville1: i != j)
        ctAntisymetrie:
            u[i][j] + u[j][i] == 1;
    
    forall(i in Villes_sauf_Ville1, j in Villes_sauf_Ville1, k in Villes_sauf_Ville1: i != j && j != k)
        ctTransitivite:
            u[i][j] + u[j][k] + u[k][i] <= 2;
    
    forall(i in Villes)
        x[i][i] == 0;
}
