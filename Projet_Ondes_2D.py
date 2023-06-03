#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LU3ME104 -- Projet numérique -- Projet : Equation des ondes 2D 

Ce script traite l'équations des ondes en 2D sur un carré 
avec conditions aux limites de Dirichlet et conditions initiales :
    
    u,tt - (u,xx + u,yy) = f pour (x,y) dans [0,1]x[0,1] et t>=0
    u = g                    pour (x,y) sur le bord  et t>=0
    u(x,y,t=0) = u0(x,y)
    u,t(x,y,t=0) = v0(x,y)

Le problème est discrétisé en espace sur un maillage régulier à NxN intervalles
0 = x_0, x_1, ... x_N = 1  et 0 = y_0, y_1, ... y_N = 1 
Les dérivées sont approchées par différences finies centrées. 
Une numérotation globale verticales par verticales est choisie.
 
On obtient un système d'équations aux dérivées ordinaires :
    
    U,tt(t) + A . U(t) = F(t) (avec CI)
  
Ce système est discrétisé en temps en utilisant également des différences finies
centrées en temps : on obtient le schéma explicite à deux pas dit "saute-mouton "
U(n+1) = [2I - dt**2 A] . U(n) - U(n-1) + dt**2 F(n)

Ce schéma est stable si la condition CFL est satifaite :

    dt < C h, avec C = 1/sqrt(2) ~~ 0.707 pour l'approximation d'ordre 2 en espace et en temps

Ce script traite le cas d'un f agité sous forme sinusoïdale et amplifier par une variation de profondeur.
(mouvement induit uniquement par l'écart initial à l'équilibre)
Ci-dessous, on
 - propose trois types de conditions initiales 
 - introduit la discrétisation en espace 
 - met en place le schéma saute-mouton 
 - sauve la solution sous la forme d'une animation

@author: J. Govare & A. Schwager
@email: jules.govare@etu.sorbonne-universite.fr antoine.schwager@etu.sorbonne-universite.fr
"""

# %% Importation de bibliothèques et paramètres généraux

import numpy as np               # Bibliothèque de calcul numérique
import matplotlib.pyplot as plt  # Bibliothèque de tracés de courbes
from matplotlib.animation import FuncAnimation 

# Les deux lignes ci-dessous imposent une police de caractère LaTeX dans les figures.
# Elles ne fonctionnent pas dans tous les systèmes d'exploitation
# params = {'text.usetex': True, 'font.family': 'sans-serif'} 
# plt.rcParams.update(params) 

#%% Paramètres du problème continu

# Configuration de référence : on ne définit que la condition initiale
my_CI = "vagues" # "zero", "stat", "bosse", "vagues" : voir la fonction ci-dessous

print('*** Problème continu : ')
print('Choix de conditions initiales : {:s}'.format(my_CI))

# Fonction qui définit les conditions initiales
def conditions_initiales(x,y,type_CI="zero",\
                         a1=3./8.,b1=5./8.,a2=3./8.,b2=5./8. ):
    """ Renvoie la position initiale u0(x,y) et la vitesse initiale v0(x,y) 
    /!\ Ici u0 satisfait u0(0) = 0 sur tout le bord du carré [0,1]x[0,1]
    Entrée : 
        x,y : vecteurs de positions (np.array de taille N)
        type_CI : type de condition initiales, choix possible parmi
            "zero" (defaut) : position u0 et vitesses v0 nulles  
            "stationnaire" : initialisation d'une onde stationnaire : 
                     u0 = sin(3\pi x)sin(4\pi y) et v0 = 0                             
            "bosse" : u0 avec une "bosse" au milieu du carré, v0 = 0  
            "vagues" : position u0 et vitesses v0 nulles 
        (a1,b1,a2,b2) : zone choisie pour la CI "bosse"
                        (defaut : [3/8,5/8]x[3/8,5/8] : centre du carré)            
        
    Sorties :        
        u0, v0 : vecteurs des positions et vitesses initiales
    """
    Nx = x.size
    Ny = y.size
    if type_CI == "zero":
        u0 = np.zeros(Nx*Ny)
        v0 = np.zeros(Nx*Ny) 
    elif type_CI == "stationnaire": 
        # Construction d'une condition 2D comme produit tensoriel de deux conditions 1D
        ux = np.sin(3*np.pi*x)
        uy = np.sin(4*np.pi*y)
        u0 = np.tensordot(ux,uy,axes=0)       # définit u0 partout dans le carré            
        u0 = u0.reshape(Nx*Ny)                # transforme u0 en vecteur (numérotation verticale par verticale)
        v0 = np.zeros(Nx*Ny)    
    elif type_CI == "bosse":        
        # Construction d'une condition 2D comme produit tensoriel de deux conditions 1D
        ux = np.cos(4*np.pi*x)**2
        uy = np.cos(4*np.pi*y)**2
        u0 = np.tensordot(ux,uy,axes=0)      # définit u0 partout dans le carré
        u0[x<a1,:] = 0                       # tronque à gauche de la bosse
        u0[x>b1,:] = 0                       # tronque à droite de la bosse
        u0[:,y<a2] = 0                       # tronque en-dessous de la bosse
        u0[:,y>b2] = 0                       # tronque au-dessus de la bosse
        u0 = u0.reshape(Nx*Ny)               # transforme u0 en vecteur (numérotation verticale par verticale)
        v0 = np.zeros(Nx*Ny)            
    elif type_CI == "vagues":
        u0 = np.zeros(Nx*Ny)
        v0 = np.zeros(Nx*Ny) 
    else:
        print("/!\ Type de conditions initiales inconnu, affecte la valeur par défaut.")
        u0, v0 = conditions_initiales(x)
        
    return u0,v0      
     
        
#%% Discrétisation en espace et définition de la matrice A
N = 100              # nombre d'intervalles (j = 0 ... N donc (N+1) points)
h = 1./N            # pas d'espace
Ntot = (N-1)**2     # taille totale du système
g = 9.81            # gravité
prof = 1               # profondeur
omg = 2           # pulsation imposée aux vagues
print('*** Discrétisation en espace : ')
print('N = {:d}'.format(N))
print('h = {:.2e}'.format(h))
print('Nombre de noeuds intérieurs Ntot = {:d}'.format(Ntot))

# Maillage régulier avec la fonction np.linspace (même maillage dans les deux directions)
x = np.linspace(0,1.,N+1) 
y = x

# Définition de A par sous-blocks
A1 = (1/h**2) * ( - np.eye(N-1,k=-1) + 4*np.eye(N-1) - np.eye(N-1,k=1) ) # T dans l'énoncé
A2 = (1/h**2) * ( - np.eye(N-1) )                                        # D dans l'énoncé

A = np.zeros((Ntot,Ntot))

for i in range(N-1):        # remplissage ligne par ligne 
    pm = (i-1)*(N-1)
    p = i*(N-1)
    q = (i+1)*(N-1)
    qp = (i+2)*(N-1)
    
    A[p:q,p:q] = A1         # bloc diagonal
    
    if i == 0:              # première ligne
        A[p:q,q:qp] = A2    # bloc sur-diagonal
    elif i == (N-2):        # dernière ligne        
        A[p:q,pm:p] = A2    # bloc sous-diagonal            
    else:
        A[p:q,pm:p] = A2    # bloc sous-diagonal            
        A[p:q,q:qp] = A2    # bloc sur-diagonal
            


#%% Schéma temporel 

tf_cible = 3.                    # temps final souhaité
if my_CI == "stat": tf_cible = 2./5.  # temps d'une période pour le cas stationnaire

D = 0.5
dt = D*h                         # /!\ Condition CFL à respecter : dt = D h < C h (donc D < C)
                                  # pour ce script, C = 1/sqrt(2) ~~ 0.707
Nt = int(np.ceil(tf_cible/dt)+1)  # Nombre de pas de temps pour approcher le temps final cible
                                  # np.ceil pour retenir l'entier supérieur
tf = (Nt-1)*dt                    # temps final réel : dernier pas de temps

print('*** Discrétisation en temps : ')
print("tfinal = {:.2g}".format(tf))
print('dt = {:.2e}'.format(dt))
print('Nt = {:d}'.format(Nt))

# Matrice  B = [2I - dt**2 A]
B = 2*np.eye(Ntot) - dt**2 * A

# Initialisation : conditions initiales appliquées aux noeuds intérieurs
u0, v0 = conditions_initiales(x[1:-1],y[1:-1],type_CI=my_CI)

# Second membre 
F = np.zeros((Nt,Ntot)) # Initialisation de la matrice F
t = 0                   # Initialisation du temps à t0
prof1 = prof+0.5        # Mise en place d'un facteur multipliant pour faire varier la profondeur
F[0,:] = 0                  # Initialisation de la force à t0 = 0

for i in range(1,Nt-1):           # Parcours du temps entre 1 et Nt
    t = t + dt                      # Evolution du temps réel t
    F[i,0] = np.sin(omg*t)*g*p      # Initialisation de la force exercée en y = 0 (CL)
    start = 206                     # Début de notre obstacle en y = 6 + j.N-1
    end = 210                       # Fin de notre obstacle en y = 10 + j.N-1
    for j in range (1,Ntot):        # Parcours de l'espace entre 1 et Ntot grâce à un seul indice 
        if j == start:              
            F[i,j] = F[i-1,j-1]*prof1       # Si j atteint le début de l'obstacle on applique le facteur multipliant de la profondeur
        elif (j > start) and (j < end) :   
            prof1 = prof1 + 0.05            # Si j sur l'obstacle on augmente progressivement la hauteur
            F[i,j] = F[i-1,j-1]*prof1
        elif j == end:          
            F[i,j] = F[i-5,j-5]             # Si j atteint la fin de l'osbtacle on récupère la valeur de notre second membre sans le facteur multipliant de la profondeur
            prof1 = prof+0.5                # Réinitialisation de la profondeur de l'obstacle à son départ
            
            if end+N-1 > Ntot:              
                start = 6                   # Réinitialisation de la valeur des débuts et fin de l'obstacle pour t+1
                end = 9
            else:
                start += N-1                # Nouvelle valeur de l'obstacle qui suit la discrétisation spatiale
                end += N-1
        else : 
            F[i,j] = F[i-1,j-1]  
            

# Initialisation et valeurs aux deux premiers temps
U = np.zeros((Nt,Ntot))         # tableau à deux dimensions 
                                # chaque ligne représente un temps
U[0,:] = u0
U[1,:] = (B/2).dot(u0) + dt*v0 + (dt**2/2) * F[0,:]

# Itérations sur tous les n
for n in range(1,Nt-1):
    U[n+1,:] = B.dot(U[n,:]) - U[n-1,:] + dt**2 * F[n,:]

#%% Critère d'erreur 
print("*** Evaluation de la solution numérique")

def err_periode(Nt):
    """Renvoie l'erreur commise au temps finale sur une période.
    Entrée :
        Nt : Nombre de pas de temps pour approcher le temps final cible
    """

    Et = np.linalg.norm(U[Nt-1,:]-U[int(np.ceil(t/dt)),:])/np.linalg.norm(U[int(np.ceil(t/dt)),:])

    return  Et

print("Erreur périodique : "+str(err_periode(Nt)))

#%% Tracé de figure
print("*** Tracé des figures")

# Paramètres des figures
my_figsize = (8,6)
FS = 18 # Taille de fonte des graduation
FSt = 22 # Taille de fonte des textes (titre, légende, axes)
myticks = np.array([0,0.5,1]) # graduations sur le carré

# Fonction pour ajouter les valeurs extremales à une solution
def Utot_2D(U):
    """ Incorpore les valeurs limites (nulles) et renvoie un tableau 2D
    Entrées :
        U : vecteur des valeurs intérieures en numérotation verticale
    Sortie :
        Utot : tableau (N+1)x(N+1) de toutes les valeurs
    """
    N = int(np.sqrt(U.size)) + 1
    U_2D = U.reshape((N-1,N-1))    
    Utot = np.zeros((N+1,N+1)) # Crée un tableau nul partout
    Utot[1:-1,1:-1] = U_2D     # Remplit les valeurs intérieures

    return Utot

# *** Tracé de la position initiale

plt.figure(1,figsize=my_figsize) 
plt.clf()
plt.pcolormesh(x,y,Utot_2D(u0).T,cmap="inferno", shading ='gouraud') # ne pas oublier le transpose !
plt.axis('square');  
cbar = plt.colorbar(); cbar.ax.tick_params(labelsize=FS) 
plt.xlim(0,1); plt.xticks(myticks,fontsize=FS)
plt.ylim(0,1); plt.yticks(myticks,fontsize=FS)
plt.tight_layout() 
plt.title("Condition initiale : $u_0$", fontsize=FSt)
plt.tight_layout()

savefig = 0
if savefig: plt.savefig('Projet_Ondes_u0_N{:d}.pdf'.format(N), bbox_inches='tight')

# *** Tracé de la position au pas de temps nplot
nplot = Nt-1 # nplot=Nt-1 pour la position finale
affiche_titre = 1 # pour afficher le titre avec des infos sur la discrétisation
plt.figure(2,figsize=my_figsize) 
plt.clf()
plt.pcolormesh(x,y,Utot_2D(U[nplot,:]).T,cmap="inferno", shading ='gouraud') # ne pas oublier le transpose !
plt.axis('square');  
cbar = plt.colorbar(); cbar.ax.tick_params(labelsize=FS) 
plt.xlim(0,1); plt.xticks(myticks,fontsize=FS)
plt.ylim(0,1); plt.yticks(myticks,fontsize=FS)
plt.tight_layout() 

if affiche_titre:
    plt.title("$u(x,y,t)$ pour $t=${:.2g}\nDiscretisation : $N={:d}$, $h={:.2g}$, $\delta t = {:.2g}$".format(nplot*dt,N,h,dt),\
          fontsize=FSt)
plt.tight_layout()    

savefig1 = 0
if savefig1: plt.savefig('Projet_Ondes_n{:d}_N{:d}.pdf'.format(nplot,N), bbox_inches='tight')

# %% Animation
print("*** Animation")

animation = 1 # initialement à zéro pour ne pas prendre trop de temps, 
              # passez à 1 et exécutez la cellule pour voir l'animation  
if animation:
    
    # Bornes des valeurs dans l'animation
    umin = -30. 
    umax = 40.
    if my_CI == "bosse": # dans le cas de la bosse on tronque les valeurs dans [-0.2, 0.2]
        umin = -0.2 
        umax = 0.2
    
    fig = plt.figure(10,figsize=my_figsize) 
    plt.clf()
    # ne pas oublier le transpose !
    image = plt.pcolormesh(x,y,Utot_2D(U[0,:]).T,\
                           cmap="inferno", shading ='gouraud', vmin=umin, vmax=umax) 
    plt.axis('square');  
    cbar = plt.colorbar(ticks=[umin,0,umax]); cbar.ax.tick_params(labelsize=FS) 
    montitre = r'Pas de temps $n = {:d}$, temps $t_n  = {:.2g}$'.format(0,0*dt)
    plt.title(montitre, fontsize=FSt) # Mise à jour du titre
    plt.xlim(0,1); plt.xticks(myticks,fontsize=FS)
    plt.ylim(0,1); plt.yticks(myticks,fontsize=FS)    
    plt.tight_layout() 
    
    # 2/ Animation avec la fonction "FuncAnimation" qui gère la mise à jour de la figure
    # cf https://pyspc.readthedocs.io/fr/latest/05-bases/12-animation.html
    dt_ms = dt * 1000 # intervale de temps entre chaque mise à jour, en ms  
    
    def animate(n):
        """
         Fonction principale définissant tout ce qui évolue dans la figure
        d'un pas de temps à un autre """
        u2D = Utot_2D(U[n,:]).T     # ne pas oublier le transpose
        u1D = u2D.ravel()           # transforme u (array 2D) en array 1D
        image.set_array(u1D)        # set_array prend un vecteur 1D en entrée d'où ravel()    
        montitre = r'Pas de temps $n = {:d}$, temps $t_n  = {:.2g}$'.format(n,n*dt)
        plt.title(montitre, fontsize=FSt) # Mise à jour du titre
        return image
         
    anim = FuncAnimation(fig, animate, interval=dt_ms, frames=Nt, repeat=False)
    plt.draw()
    plt.show()    
    
    # Sauvegarde en film (format MP4)
    saveanim = 0 # passer à 1 pour sauvegarder
    if saveanim:
        anim.save("Onde_{:s}_2D.mp4".format(my_CI), fps=50)	

print("FIN DU PROGRAMME")