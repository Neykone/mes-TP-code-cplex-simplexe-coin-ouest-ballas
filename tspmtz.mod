/*********************************************
 * OPL 22.1.1.0 Model
 * Author: HP
 * Creation Date: 17 nov. 2025 at 17:08:04
 *********************************************/

 int n=...;
 range ville=1..n;
 range villesaufville1=2..n;
 
 float d[ville][ville]=...;
 dvar int u[ville] in ville ;
 
 dvar boolean x[ville][ville];
 
 minimize
    sum(i in ville, j in ville) d[i][j] * x[i][j];
    
    
subject to {
  
  forall(i in ville)
    sum(j in ville) x[i][j]==1;
    
  forall(j in ville)
    sum(i in ville) x[i][j]==1;  
    
  forall(i in villesaufville1,j in villesaufville1)  
      if(i != j)
      u[i] - u[j] + (n-1)*x[i][j] <= n - 2 ;
      
      forall(i in ville)
        u[i] >=1;
      forall(i in ville) 
        u[i] <=n;
        
  forall(i in ville) 
     x[i][i]==0;   

  
}    
    
    