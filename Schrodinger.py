"""This package in intended for the resolution of the 1D stationnary Schrödinger equation with the Numerov scheme and the mid-point matching technique The energy is given in Hartree, while the distance is given in Bohr radius"""
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

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
    """
    This function applies the numerov scheme on a discretized 1D space represented by x_range, in order to produce an approximate solution of the Schrödinger equation with the potential V and the energy E.
    
    psi_range = array which will contain the psi_i, with the boundary values already initialised
    x_range = array containing the values of x
    V = function for the potential
    E = Energy to be tested
    direction -> 1 = increasing x ; -1 = decreasing x
    i_start = index of the first element of psi_range to be calculated by the numerov scheme
    
    return : an array containing the approximate solution psi(x) for each x of x_range"""
    
    if (len(psi_range) != len (x_range)):
        raise Exception("psi_range and x_range must be of the same length, but they are not")
    
    #useful quantities for the scheme
    Q = lambda x : 2*(E-V(x))
    h = x_range[1]-x_range[0]  
    psi_out = psi_range.copy()
    
    #P_window : order of magnitude allowed for the calculated values
    P_window = 5
    
    #execution of the Numerov scheme
    if direction == 1 : #for increasing x
        for i in range(i_start,len(psi_range)):
            psi_out[i] = (2*(1-5/12*h**2*Q(x_range[i-1]))*psi_out[i-1]-(1+1/12*h**2*Q(x_range[i-2]))*psi_out[i-2])/(1+1/12*h**2*Q(x_range[i]))
            
            #control of the amplitude of the wave function
            if abs(psi_out[i]) > 10**P_window:
                for k in range(i+1):
                    psi_out[k] = psi_out[k]/10
            
            if abs(psi_out[i]) < 10**(-P_window):
                for k in range(i+1):
                    psi_out[k] = psi_out[k]*10
            
    if direction == -1 : #for decreasing x
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
    """
    This function applies the mid_point matching technique in order to apply both boundary conditons (left and right) on the wave function. The logarithmic  derivative error at the matching point is calculated. It handles the problematic case of having a matching point on a node, where the logarithmic derivative diverges, by translating the matching point.
    
    psi_range = array which will contain the psi_i, with the boundary values already initialised
    x_range = array containing the values of x
    V = function for the potential
    E = Energy to be tested
    N_x_c = index of the mid_point in the x_range
    
    return : the logarithmic derivative error,
    the number of nodes of the solution,
    an array containing the approximate solution psi(x) for each x of x_range"""
    
    #useful quantity for the scheme
    h = x_range[1]-x_range[0]
    
    #cutting of the ranges at the midpoint
    x_left = x_range.copy()[0:N_x_c+1]
    x_right = x_range.copy()[N_x_c:] 
    psi_left = psi_range.copy()[0:N_x_c+1]
    psi_right = psi_range.copy()[N_x_c:]
    
    
    #performing the left and right numerov schemes
    psi_left = numerov(psi_left,x_left,V,E,1)
    psi_right = numerov(psi_right,x_right,V,E,-1)
    
    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    #### Handling of the case of a diverging logarithmic derivative
    
    #Checking if the inverse of the logarithmic derivative is too small (i.e. the logarithmic derivative is too big)
    seuil = 10**-2 #size of the test window
    test_1 = -seuil < (psi_left[-1]*h)/(psi_left[-1]-psi_left[-2]) < seuil
    test_2 = -seuil < (psi_right[0]*h)/(psi_right[1]-psi_right[0]) < seuil
    
    #flag for the convergence of the procedure:
    converged = False
    
    #saving the already found psi
    old_psi_left = psi_left 
    old_psi_right = psi_right
    
    #translating x_c to the right if the inverse of the logarithmic derivative is too small
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
        
    #if still not converged, translating x_c to the left
    if not converged :
        #resetting x_c to the intial value
        psi_left = old_psi_left
        psi_right = old_psi_right
        x_left = x_range.copy()[0:N_x_c+1]
        x_right = x_range.copy()[N_x_c:] 
        
        #translation to the left
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
    
    #if still not converge, abortion of the procedure
    if not converged :
        print("In Do_mid_point : matching error cannot be evaluated for E= ",E,"and N_x_c = ",N_x_c)
        return np.NaN , np.NaN , np.NaN
    #-------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    
            
    #junction of the two pieces of th ewave function :
    psi_out = np.concatenate((psi_left[0:-1],psi_right*((psi_left[-1])/(psi_right[0]))))
    
    #evaluation of the logarithmic error
    log_error = 2*(psi_left[-1]-psi_left[-2])/((psi_left[-1])*h) - 2*(psi_right[1]-psi_right[0])/((psi_right[0])*h)

    #evaluation of the number of nodes of the wave function
    N = 0
    for i in range(1,len(psi_out)) :
        if (psi_out[i]*psi_out[i-1] < 0):
            N += 1
    
    return log_error, N , psi_out

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def slice_E_arr(psi_range,x_range,V,E_arr,N_x_c,slices,temp_slices,remove_borders = False):
    """
    Function used to slice E_arr in slices containing only one number of nodes for the associated wavefunction.
    The slices which succesfuly contains only one number of nodes are stored in "slices"
    The slices which cannot be cut small enough to contain only one number of nodes are stored in "temp_slices", either to be discarded or to run them again in the function after having reduced their energy step.
    "remove_borders" has to be set to True when E_arr is a temp_slice coming from a precedent call of the function
    
    psi_range = array which will contain the psi_i, with the boundary values already initialised
    x_range = array containing the values of x
    V = function for the potential
    E_arr = array containing the energies to be tested
    N_x_c = index of the mid_point in the x_range
    slices = array to which the succesful slices will be appended
    temp_slices = array to which the unsuccesful slices will be appended
    remove_borders = if set to True, the function will ignore the slices found at the borders
    
    return : a boolean : True if the procedure did work / False if it did not work"""
    
    #verifying that the length of the E_arr is sufficient
    if len(E_arr)<10:
        print("IN FUNCTION slice_E_arr : insuficient length of E_arr")
        return False
    
    #retrieving the number of nodes for each energies in E_range with the mid-point matching technique
    N_arr = np.zeros(len(E_arr))
    for i in range(len(E_arr)):
        log_error , N_arr[i] , psi_out = Do_mid_point(psi_range,x_range,V,E_arr[i],N_x_c)
     
    #finding where the number of nodes changes, and how much
    E_crit = [] #array which will contains tuples (index of the change, size of the change)
    for i in range(0,len(N_arr)-1) :
        if N_arr[i+1]-N_arr[i] > 0 :
            E_crit.append((i,N_arr[i+1]-N_arr[i]))
    
    #if there is no change in the number of nodes, do nothing
    if len(E_crit)==0 :
        return True
 
    #for the left boundary:
    if (not remove_borders):
        #append to slices the slice running from the boundary to the first change of number of nodes
        slices.append(( [ E_arr[0] , E_arr[np.min([E_crit[0][0]+2,len(E_arr)-1])] ] ,N_arr[np.min([E_crit[0][0],len(N_arr)-1])]))
        if E_crit[0][1] > 1 :
            #if the change of number of nodes is greater than 1, append to slices a slice around the position of the change
            temp_slices.append([ E_arr[E_crit[0][0]] , E_arr[np.min([E_crit[0][0]+2,len(E_arr)-1])] ])
    
    #for the energies in between
    for i in range(1,len(E_crit)) :
        #append to slices the slice running from one change of number of nodes to the other
        slices.append(( [ E_arr[E_crit[i-1][0]] , E_arr[np.min([E_crit[i][0]+2,len(E_arr)-1])] ],N_arr[np.min([E_crit[i][0],len(N_arr)-1])]))
        if E_crit[i][1] > 1 :
            #if the change of number of nodes is greater than 1, append to slices a slice around the position of the change
            temp_slices.append([ E_arr[E_crit[i][0]] , E_arr[np.min([E_crit[i][0]+2,len(E_arr)-1])] ])

    #handling the case of the first change of number of nodes when we don't want the boundaries.
    if (remove_borders and E_crit[0][1] > 1):
        temp_slices.append([ E_arr[E_crit[0][0]] , E_arr[np.min([E_crit[0][0]+2,len(E_arr)-1])] ])
    
    #for the right boundary:
    if (not remove_borders):
        slices.append(( [ E_arr[E_crit[-1][0]] , E_arr[-1]],N_arr[-1]))
        
    return True

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def find_E_N(psi_range,x_range,V,E_min,E_max,N_x_c,N,err=10**-2):
    """Function used to perform a dichotomy on slices comming from "slice_E_arr" to find the energy minimizing the logarithmic derivative error in the mid-point matching technique
    
    psi_range = array which will contain the psi_i, with the boundary values already initialised
    x_range = array containing the values of x
    V = function for the potential
    E_min / E_max = boudaries of the energy slice
    N_x_c = index of the mid_point in the x_range
    N = expected number of nodes for the solution to be found
    err = precision at which the dichotomy will stop
    
    return : the energy which minimizes the logarithmic derivative error"""
    
    k = 20 #number of test point for each step of the dichotomy
    
    #declaration of useful variables
    err_arr = np.zeros(k)
    N_arr = np.zeros(k)
    E_m = E_min
    E_M = E_max
    E_guess = E_min
    
    #while the dichotomy scope is larger than err, continue the dichotomy
    while (np.abs(E_m - E_M) > err):
        
        #creation of the test points
        E_arr = np.linspace(E_m,E_M,k)
        
        #evalutation of the error at each test points
        for i in range(k):
            err_arr[i] , N_arr[i] , psi_out = Do_mid_point(psi_range,x_range,V,E_arr[i],N_x_c)
        
        #retrieving the test point which minmizes the logarithmic derivative error
        n_E_guess = np.argmin(np.abs(err_arr))
        
        #if the number of nodes of the wave function with energy E_guess is the expected, we shrink the scope of the dichotomy
        if N_arr[n_E_guess] == N :
            E_guess = E_arr[n_E_guess]
            E_m = E_arr[np.max([n_E_guess-1,0])]
            E_M = E_arr[np.min([n_E_guess+1,k-1])]
        
        #if the number of nodes of the wave function with energy E_guess isn't the expected, we cut the initial scope at this value and restart the dichotomy
        elif N_arr[n_E_guess] > N:
            E_M = E_arr[np.max([n_E_guess-1,0])]
        elif N_arr[n_E_guess] < N:
            E_m = E_arr[np.min([n_E_guess+1,k-1])]
            
    return E_guess

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def normalization(psi_range, x_range):
    """function used to normalise a wave function
    
    psi_range = array containing the values of the wave function for each points in x_range
    x_range = array containing the values of x
    
    return : an array containing the values of the normalised wave function for each points in x_range """
    
    norm = sp.simps(abs(psi_range)**2,x_range) #evaluation of the normalisation factor
    return psi_range/np.sqrt(norm)

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

def resolution(V,E_min,E_max,x_m,x_M,x_c,dx=-1,eps=10**-5,err=10**-3):
    """Function used to find numerical approximation of the eingenvalues of the stationnary Schrödinger equation for a confining potential, using the Numerov scheme and the mid-point matching technique. Vanishing boundary conditions are used. 
    In order to check the validity of the solutions, the scalar product beetween all two different wave functions found is performed, and the maximum value obtained is displayed. This value should be close to zero as solutions of the Schrödinger equation should be orthonogonal for different energies.
    The potential V is graphed using pyplot, as well as the wave functions shifted to their corresponding energy. Althought this representation bears no physical meaning, it summarises all the information found by the procedure.
    
    V = function for the potential
    E_min / E_max = boundaries of the allowed energy range for the wave functions, care should be taken to set a range which only allows bound states
    x_m / x_M = boundaries of the allowed space range for the wave function, the expected wave functions should be close to zero at those points.
    x_c = x value of the matching point for the mid-point matching technique (should be preferably set at a value where the amplitude of the expected
    wave functions will be maximal, i.e. at the position of the minimum for potential wells)
    dx = step for the discretisation of the space
    eps = initialisation value of the wave functions
    err = precision requested on the energy
    
    return : an array containing for each solution found a tuple (E,N), with E being the numerical approximation of the energy which has been found and N the number of nodes of the corresponding wave function"""
    
    ##--------------------------------------------------------------
    ##--------------------------------------------------------------
    #checking the validity of the arguments
    args_ok = isinstance(V, types.FunctionType) and (E_min < E_max) and (x_m < x_c <x_M)
    if (not args_ok) :
        print('IN FUNCTION resolution : arguments are not valid')
        return 0
    
    
    ##--------------------------------------------------------------
    ##--------------------------------------------------------------
    #creation of the initialisations sets.
    
    print('Creation of initial sets')
    
    if dx == -1 : dx = (x_M-x_m)/100 #setting a value by default
    
    x_range = np.linspace(x_m,x_M,int((x_M-x_m)/dx)) #array containing the discretized space

    #creation and initialisation of the wave-function's array.
    psi_range = np.zeros(int((x_M-x_m)/dx)) 
    psi_range[1] = eps
    psi_range[-2] = eps
    
    #calculation of the x_range index of x_c
    N_x_c = int((x_c-x_m)/dx)
        
    print("Mid-point matching set at : x =", (x_M-x_m)*(N_x_c/len(x_range))+x_m)
    
    ##--------------------------------------------------------------
    ##--------------------------------------------------------------
    #slicing [E_min,E_max] in arrays with constant number of nodes.
    
    print('\n-Finding energy ranges with fix number of nodes :')
    
    slices = [] #array containing (slice,N in slice)
    temp_slices = [] #array containing the slices that still needs to be processed
    
    #first search of energy where the number of nodes changes  
    k=2*10**1 #number of test points for the search
    E_arr = np.linspace(E_min,E_max,k) #array of energy to be sliced
    
    #execution of the slice_E_arr dunction to slice E_arr
    slice_E_arr(psi_range,x_range,V,E_arr,N_x_c,slices,temp_slices) 
    
    converged = False #flag for the convergence of the ongoing procedure
    
    sys.stdout.write("Number of ranges found so far : "+str(len(slices)))
    size_of_len_slices = int(np.log10(max(len(slices),1)))+1 #number of decimals needed in order to display len(slices)
    
    for i in range(30):
        
        #actualising the number of slices found so far
        sys.stdout.write(size_of_len_slices*"\b"+str(len(slices)))
        sys.stdout.flush()
        size_of_len_slices = int(np.log10(max(len(slices),1)))+1
        
        #re execution of slice_E_arr on the unsuccesful slices stored in temp_slices
        if (len(temp_slices) != 0):
            for slc in temp_slices.copy():
                E_arr = np.linspace(slc[0],slc[-1],k)
                temp_slices.pop(0)
                slice_E_arr(psi_range,x_range,V,E_arr,N_x_c,slices,temp_slices,remove_borders = True)
        else :
            #if there is no more unsuccesful slices, stop the ongoing procedure
            converged = True
            sys.stdout.write("\n")
            break
            
    #aborting the function run if the slicing is unsuccesful
    if converged == False :
        print("IN FUNCTION resolution : cannot have a convergence on the slicing of the energies")
        return False
    
    ##--------------------------------------------------------------
    ##--------------------------------------------------------------
    #retrieving the Energie minimizing the log_error for each slice
    E_sol = []
    
    print("\n-Performing dichotomy on each energy ranges to find solutions :")
    
    #executing the find_E_N function on each slices to find energies minimizing the logarithmic derivative error
    for slc in tqdm(slices):
        E_guess = find_E_N(psi_range,x_range,V,slc[0][0],slc[0][1],N_x_c,slc[1],err=err)
        
        #storing the given value if it's not one of the boundary
        if not (E_guess == E_min or E_guess == E_max):
            E_sol.append(E_guess)
        else :
            print("Ignoring an ambiguous solution : energy of the solution is one of the boundaries")
            
    ##--------------------------------------------------------------
    ##--------------------------------------------------------------    
    #formating and plotting the data
    
    print("\n-Formating the results:")
    
    E_sol = np.asanyarray(E_sol)
    E_sol.sort() #sorting the energies in ascending order
    
    #declaration of useful variables
    E_out = [] #array to be given as output
    a_psi_out = []
    b_psi_out = []
    
    #retriving the number of nodes and the wave functions for each energies in E_sol
    for E_found in tqdm(E_sol) :
        err_psi , N_psi , psi_out = Do_mid_point(psi_range,x_range,V,E_found,N_x_c)
        
        psi_out = normalization(psi_out,x_range)
        a_psi_out.append(psi_out+E_found)
        b_psi_out.append(psi_out)
        
        E_out.append((E_found,N_psi)) #formating the output
    
    
    ##--------------------------------------------------------------
    ##--------------------------------------------------------------    
    #checking the orthogonality of the solutions
    
    print("\n-Checking the orthogonality :")
    
    max_scalar_product = 0 #variable for storing the maximal value of the scalar products between two different 
    for i in range(len(b_psi_out)):
        for j in range(1,len(b_psi_out)-i):
            if i != len(b_psi_out)-j :
                max_scalar_product = max(sp.simps(b_psi_out[i]*b_psi_out[len(b_psi_out)-j],x_range),max_scalar_product,key=abs)
    
    print("maximum amplitude of scalar product : ",max_scalar_product)
    
    
    ##--------------------------------------------------------------
    ##--------------------------------------------------------------
    #plotting the solutions
    print("\n-Plotting the solutions :")
    plt.figure(figsize=[9,7])
    
    plt.ylabel(r'Energy ($E_H$)',fontsize=15)
    plt.xlabel(r'x ($a_0$)',fontsize=15)
    
    psi = np.array(a_psi_out)
    potentiel = V(x_range)
    psi_potentiel = np.concatenate((potentiel, psi), axis = None)
    amplitude = psi.max() - psi_potentiel.min()
    y_min = psi_potentiel.min() - abs(0.05*amplitude)
    if psi_potentiel.max() - psi.max() < amplitude/2 :
        y_max = psi_potentiel.max() + abs(0.05*amplitude)
    else :
        y_max = psi.max() + abs(0.05*amplitude)
    
    plt.ylim([y_min, y_max])
    
    plt.plot(x_range,V(x_range))
    for n in tqdm(range(len(a_psi_out))):
        plt.plot(x_range , a_psi_out[n])
    
    ##--------------------------------------------------------------
    ##--------------------------------------------------------------
    
    return E_out

