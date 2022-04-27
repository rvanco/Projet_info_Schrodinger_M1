import numpy as np
from matplotlib import pyplot as plt
import types
import scipy.integrate as sp
from tqdm.notebook import tqdm
import sys

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def numerov(psi_range,x_range,V,E,direction,i_start=2):
    """psi_range = array contenant les psi_i avec les valeurs aux bords déjà mises en place 
    x_range = array contenant les x
    V = fonction jouant le role de potentiel
    E = energie à tester
    direction -> 1 = gauche/droite  -1=droite/gauche
    i_start = indice du prmeier élément à être traité par la procédure"""
    
    if (len(psi_range) != len (x_range)):
        raise Exception("psi_range and x_range must be of the same length, but they are not")
    
    #création de variables utiles
    Q = lambda x : 2*(E-V(x))
    h = x_range[1]-x_range[0]  
    psi_out = psi_range.copy()
    
    #paramètre
    P_window = 4
    
    #réalisation de l'intégration de Numerov
    if direction == 1 :
        for i in range(i_start,len(psi_range)):
            psi_out[i] = (2*(1-5/12*h**2*Q(x_range[i-1]))*psi_out[i-1]-(1+1/12*h**2*Q(x_range[i-2]))*psi_out[i-2])/(1+1/12*h**2*Q(x_range[i]))
            
            if abs(psi_out[i]) > 10**P_window:
                for k in range(i+1):
                    psi_out[k] = psi_out[k]/10
            
            if abs(psi_out[i]) < 10**(-P_window):
                for k in range(i+1):
                    psi_out[k] = psi_out[k]*10
            
    if direction == -1 :
        for i in range(len(psi_range)-1-i_start,-1,-1):
            psi_out[i] = (2*(1-5/12*h**2*Q(x_range[i+1]))*psi_out[i+1]-(1+1/12*h**2*Q(x_range[i+2]))*psi_out[i+2])/(1+1/12*h**2*Q(x_range[i]))
            
            if abs(psi_out[i]) > 10**P_window:
                for k in range(i,len(psi_range)):
                    psi_out[k] = psi_out[k]/10
            
            if abs(psi_out[i]) < 10**(-P_window):
                for k in range(i,len(psi_range)):
                    psi_out[k] = psi_out[k]*10
                    
    return psi_out

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def Do_mid_point (psi_range,x_range,V,E,N_x_c):
    """psi_range = array contenant les psi_i avec les valeurs aux bords déjà mises en place 
    x_range = array contenant les x
    V = fonction jouant le role de potentiel
    E = energie à tester
    N_x_c = indice du mid-point"""
    
    h = x_range[1]-x_range[0]
    
    #découpage des ranges au mid-point
    x_left = x_range.copy()[0:N_x_c+1]
    x_right = x_range.copy()[N_x_c:] 
    
    psi_left = psi_range.copy()[0:N_x_c+1]
    psi_right = psi_range.copy()[N_x_c:]
    
    
    #réalisation des intégrations de numerov
    psi_left = numerov(psi_left,x_left,V,E,1)
    psi_right = numerov(psi_right,x_right,V,E,-1)
    
    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    ####gestion du cas où x_c est proche d'un noeud :
    seuil = 10**-2
    test_1 = -seuil < (psi_left[-1]*h)/(psi_left[-1]-psi_left[-2]) < seuil
    test_2 = -seuil < (psi_right[0]*h)/(psi_right[1]-psi_right[0]) < seuil
    
    #flag pour la convergence de la procédure :
    converged = False
    
    #sauvegarde des psi
    old_psi_left = psi_left 
    old_psi_right = psi_right
    
    #décalage à droite
    for i in range(int(len(x_range)/25)):
        if test_1 or test_2 :
            x_left = np.concatenate((x_left,[x_range[len(x_left)]]))
            x_right = x_right[1:]
            psi_left = numerov(np.concatenate((psi_left,np.zeros(1))),x_left,V,E,1,i_start=len(psi_left))
            psi_right = psi_right[1:]
        else :    
            converged = True
            break
        test_1 = -seuil < (psi_left[-1]*h)/(psi_left[-1]-psi_left[-2]) < seuil
        test_2 = -seuil < (psi_right[0]*h)/(psi_right[1]-psi_right[0]) < seuil
        
    #si pas convergé, décalage à gauche
    if not converged :
        #récupération des anciens psi
        psi_left = old_psi_left
        psi_right = old_psi_right
        #récupération des anciens x_range
        x_left = x_range.copy()[0:N_x_c+1]
        x_right = x_range.copy()[N_x_c:] 
        
        #décalage à gauche
        for i in range(int(len(x_range)/25)):
            if test_1 or test_2 :
                x_left = x_left[0:-1]
                x_right = np.concatenate(([x_range[-len(x_right)-1]],x_right))
                psi_left = psi_left[0:-1]
                psi_right = numerov(np.concatenate((np.zeros(1),psi_right)),x_right,V,E,-1)
            else :    
                converged = True
                break
            test_1 = -seuil < (psi_left[-1]*h)/(psi_left[-1]-psi_left[-2]) < seuil
            test_2 = -seuil < (psi_right[0]*h)/(psi_right[1]-psi_right[0]) < seuil
    
    #si pas convergé, affichage message d'erreur
    if not converged :
        print("In Do_mid_point : matching error cannot be evaluated for E= ",E,"and N_x_c = ",N_x_c)
        return np.NaN , np.NaN , np.NaN
    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    
            
    #raccordement des deux morceaux :
    psi_out = np.concatenate((psi_left[0:-1],psi_right*((psi_left[-1])/(psi_right[0]))))
    
    #évaluation de l'erreur du raccord en x_c
    log_error = 2*(psi_left[-1]-psi_left[-2])/((psi_left[-1])*h) - 2*(psi_right[1]-psi_right[0])/((psi_right[0])*h)

    N = 0
    
    for i in range(1,len(psi_out)) :
        if (psi_out[i]*psi_out[i-1] < 0):
            N += 1
    
    return log_error, N , psi_out

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def find_change(E_arr,N_arr):
    #find where and how much the nodes (N_arr) changes w.r.t E_arr
    data = []
    for i in range(0,len(E_arr)-1) :
        if N_arr[i+1]-N_arr[i] > 0 :
            data.append((i,N_arr[i+1]-N_arr[i]))
    return data

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def slice_E_arr(psi_range,x_range,V,E_arr,N_x_c,slices,temp_slices,remove_borders = False):
    """Function used to slice E_arr in slices containing only one number of nodes (except on the boundaries of the slices) 
    for the associated wavefunction.
    
    The slices which succesfuly contains only one number of nodes are stored in "slices"
    The slices which cannot be cut small enough to contain only one number of nodes are stored in "temp_slices", either to be discarded or to
    run them again in the function after having reduced their energy step.
    
    "remove_borders" has to be set to True when E_arr is a temp_slice coming from a precedent call of the function"""
    
    #verifying that the length of the E_arr is sufficient
    if len(E_arr)<10:
        print("IN FUNCTION slice_E_arr : insuficient length of E_arr")
        return False
    
    N_arr = np.zeros(len(E_arr))
    for i in range(len(E_arr)):
        log_error , N_arr[i] , psi_out = Do_mid_point(psi_range,x_range,V,E_arr[i],N_x_c)
    E_crit = find_change(E_arr,N_arr)
    
    if len(E_crit)==0 :
        print("IN FUNCTION slice_E_arr : No change of number of nodes found")
        return False
 
    if (not remove_borders):
        slices.append(( [ E_arr[0] , E_arr[np.min([E_crit[0][0]+2,len(E_arr)-1])] ] ,N_arr[np.min([E_crit[0][0],len(N_arr)-1])]))
        if E_crit[0][1] > 1 :
            temp_slices.append([ E_arr[E_crit[0][0]] , E_arr[np.min([E_crit[0][0]+2,len(E_arr)-1])] ])
    
    for i in range(1,len(E_crit)) :
        slices.append(( [ E_arr[E_crit[i-1][0]] , E_arr[np.min([E_crit[i][0]+2,len(E_arr)-1])] ],N_arr[np.min([E_crit[i][0],len(N_arr)-1])]))
        if E_crit[i][1] > 1 :
            temp_slices.append([ E_arr[E_crit[i][0]] , E_arr[np.min([E_crit[i][0]+2,len(E_arr)-1])] ])

    if (remove_borders and len(E_crit) == 1):
        temp_slices.append([ E_arr[E_crit[0][0]] , E_arr[np.min([E_crit[0][0]+2,len(E_arr)-1])] ])
    
    
    if (not remove_borders):
        slices.append(( [ E_arr[E_crit[-1][0]] , E_arr[-1]],N_arr[-1]))
        
    return True

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def find_E_N(psi_range,x_range,V,E_min,E_max,N_x_c,N,err=10**-2):
    k = 20
    err_arr = np.zeros(k)
    N_arr = np.zeros(k)
    E_m = E_min
    E_M = E_max
    E_guess = E_min
    while (np.abs(E_m - E_M) > err):
        E_arr = np.linspace(E_m,E_M,k)
        for i in range(k):
            err_arr[i] , N_arr[i] , psi_out = Do_mid_point(psi_range,x_range,V,E_arr[i],N_x_c)
            
        n_E_guess = np.argmin(np.abs(err_arr))
        
        if N_arr[n_E_guess] == N :
            E_guess = E_arr[n_E_guess]
            E_m = E_arr[np.max([n_E_guess-1,0])]
            E_M = E_arr[np.min([n_E_guess+1,k-1])]
        elif N_arr[n_E_guess] > N:
            E_M = E_arr[np.max([n_E_guess-1,0])]
        elif N_arr[n_E_guess] < N:
            E_m = E_arr[np.min([n_E_guess+1,k-1])]
            
    return E_guess

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def normalization(psi, x):
    norm = sp.simps(abs(psi)**2,x)
    return psi/np.sqrt(norm)

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def resolution(V,E_min,E_max,x_m,x_M,x_c,dx=-1,eps=10**-5,err=10**-3):
    ##--------------------------------------------------------------
    #checking the validity of the arguments
    args_ok = isinstance(V, types.FunctionType) and (E_min < E_max) and (x_m < x_M)
    if (not args_ok) :
        print('IN FUNCTION resolution : arguments are not valid')
        return 0
    
    
    ##--------------------------------------------------------------+

    #creation of the initialisations sets.
    
    print('Creation of initial sets')
    
    if dx == -1 : dx = (x_M-x_m)/100
    
    x_range = np.linspace(x_m,x_M,int((x_M-x_m)/dx))

    psi_range = np.zeros(int((x_M-x_m)/dx))
    psi_range[0] = psi_x_m
    psi_range[1] = psi_x_m + eps
    psi_range[-1] = psi_x_M
    psi_range[-2] = psi_x_M + eps
    
    N_x_c = int((x_c-x_m)/dx)
        
    print("Mid-point matching set at : x =", (x_M-x_m)*(N_x_c/len(x_range))+x_m)
    
    ##--------------------------------------------------------------
    #slicing [E_min,E_max] in arrays with constant number of nodes.
    
    print('\n-Finding energy ranges with fix number of nodes :')
    
    slices = [] #array containing (slice,N in slice)
    temp_slices = [] #array containing the slices that still needs to be processed
    
    #first search of energy where the number of nodes changes  
    k=2*10**1 #number of test points for the search
    E_arr = np.linspace(E_min,E_max,k)
    
    slice_E_arr(psi_range,x_range,V,E_arr,N_x_c,slices,temp_slices)
    
    converged = False
    
    sys.stdout.write("Number of ranges found so far : "+str(len(slices)))
    size_of_len_slices = int(np.log10(max(len(slices),1)))+1 #pour l'affichage
    
    for i in range(30):
        
        sys.stdout.write(size_of_len_slices*"\b"+str(len(slices)))
        sys.stdout.flush()
        size_of_len_slices = int(np.log10(max(len(slices),1)))+1
        
        if (len(temp_slices) != 0):
            for slc in temp_slices.copy():
                E_arr = np.linspace(slc[0],slc[-1],k)
                temp_slices.pop(0)
                slice_E_arr(psi_range,x_range,V,E_arr,N_x_c,slices,temp_slices,remove_borders = True)
        else :
            converged = True
            sys.stdout.write("\n")
            break
    if converged == False :
        print("IN FUNCTION resolution : cannot have a convergence on the slicing of the energies")
        return False
    ##--------------------------------------------------------------
    #retrieving the Energie minimizing the log_error for each slice
    E_sol = []
    
    print("\n-Performing dichotomy on each energy ranges to find solutions :")
    
    for slc in tqdm(slices):
        E_guess = find_E_N(psi_range,x_range,V,slc[0][0],slc[0][1],N_x_c,slc[1],err=err)
        if not (E_guess == E_min or E_guess == E_max):
            E_sol.append(E_guess)
        else :
            print("Supression of an ambiguous solution : energy of the solution is one of the boundaries")
    
    
    #formating and plotting the data
    
    print("\n-Formating the results:")
    
    E_sol = np.asanyarray(E_sol)
    E_sol.sort()
    E_out = []
    a_psi_out = []
    b_psi_out = []
    
    for E_found in tqdm(E_sol) :
        err_psi , N_psi , psi_out = Do_mid_point(psi_range,x_range,V,E_found,N_x_c)
        
        psi_out = normalization(psi_out,x_range)
        a_psi_out.append(psi_out+E_found)
        b_psi_out.append(psi_out)
        
        E_out.append((E_found,N_psi))
    
    #checking the orthogonality of the solutions
    print("\n-checking the orthogonality :")
    max_scalar_product = 0
    for i in range(len(b_psi_out)):
        for j in range(1,len(b_psi_out)-i):
            if i != len(b_psi_out)-j :
                max_scalar_product = max(sp.simps(b_psi_out[i]*b_psi_out[len(b_psi_out)-j],x_range),max_scalar_product,key=abs)
    
    print("maximum amplitude of scalar product : ",max_scalar_product)
    
    #plotting the solutions
    print("\n-plotting the solutions :")
    plt.figure()
    
    plt.ylabel('Energy')
    plt.xlabel('x')
    
    psi = np.array(a_psi_out)
    potentiel = V(x_range)
    psi_potentiel = np.concatenate((potentiel, psi), axis = None)
    amplitude = psi.max() - psi_potentiel.min()
    y_min = psi_potentiel.min() - abs(0.05*amplitude)
    y_max = psi.max() + abs(0.05*amplitude)
    
    plt.ylim([y_min, y_max])
    
    plt.plot(x_range,V(x_range))
    for n in tqdm(range(len(a_psi_out))):
        plt.plot(x_range , a_psi_out[n])
  
    
    
    return E_out