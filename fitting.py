def supersample(x,f=10):
    """x needs to be sorted more or less have a constant derivative.
    f is half of the oversampling amount.
    I checked that with f=10, the function samples effectively, around the existing values.
    Meaning that edge values are not repeated (that's what's all the +-dx1/4/f stuff is for.)"""
    import numpy as np
    x_left = x[0]-(x[1]-x[0])
    x_right = x[-1]+(x[-1]-x[-2])

    X = np.concatenate([[x_left],x,[x_right]])#same as x but padded on both sides.

    x_super_elements = []
    for i in range(1,len(X)-1):
        dx1=X[i]-X[i-1]
        dx2=X[i+1]-X[i]

        x_super_elements.append(np.concatenate([np.linspace(X[i-1]+dx1/2+dx1/4/f,X[i]-dx1/4/f,f+1),np.linspace(X[i]+dx2/4/f,X[i]+dx2/2-dx2/4/f,f+1)]))#Start halfway between x[i] and the previous value and then add small samples until x[i]

    return(np.array(x_super_elements))#This implictly reshapes to an array with size len(x) times f+2


#The model for 1 line. ALSO CHANGE IN THE NUMPY CODE BELOW. THIS ONE IS THE NON-JITTED VERSION.
def gaussian_skewed(x_super, A1=0, mu1=1.0,sigma1=1.0,alpha1=0,c0=1.0,c1=0.0,c2=0.0,c3=0.0,
                        A2=0.0,mu2=0.0,sigma2=1.0,alpha2=0.0,
                        A3=0.0,mu3=0.0,sigma3=1.0,alpha3=0.0,
                        A4=0.0,mu4=0.0,sigma4=1.0,alpha4=0.0,
                        A5=0.0,mu5=0.0,sigma5=1.0,alpha5=0.0,                        
                        A6=0.0,mu6=0.0,sigma6=1.0,alpha6=0.0,
                        A7=0.0,mu7=0.0,sigma7=1.0,alpha7=0.0,
                        A8=0.0,mu8=0.0,sigma8=1.0,alpha8=0.0,absorption=True):
            """x goes in units of wavelength. Sigma goes in km/s. Allows for up to 8 components to be set."""
            import jax.scipy.special
            import numpy as np
            c = 299792.458 #km/s
            if mu2 == 0: mu2 = mu1
            if mu3 == 0: mu3 = mu1
            if mu4 == 0: mu4 = mu1
            if mu5 == 0: mu5 = mu1
            if mu6 == 0: mu6 = mu1
            if mu7 == 0: mu7 = mu1
            if mu8 == 0: mu8 = mu1
            X1 = x_super-mu1
            X2 = x_super-mu2
            X3 = x_super-mu3
            X4 = x_super-mu4
            X5 = x_super-mu5
            X6 = x_super-mu6
            X7 = x_super-mu7
            X8 = x_super-mu8
            V1 = X1*c/mu1
            V2 = X2*c/mu2
            V3 = X3*c/mu3
            V4 = X4*c/mu4
            V5 = X5*c/mu5
            V6 = X6*c/mu6
            V7 = X7*c/mu7
            V8 = X8*c/mu8
            poly = c0 + c1*V1 + c2*V1**2 +c3*V1**3
            G1 = (np.exp(-0.5 * (V1/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V1/sigma1/np.sqrt(2)))
            G2 = (np.exp(-0.5 * (V2/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V2/sigma2/np.sqrt(2)))
            G3 = (np.exp(-0.5 * (V3/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V3/sigma3/np.sqrt(2)))
            G4 = (np.exp(-0.5 * (V4/sigma4)**2)) * (1+jax.scipy.special.erf(alpha4*V4/sigma4/np.sqrt(2)))
            G5 = (np.exp(-0.5 * (V5/sigma5)**2)) * (1+jax.scipy.special.erf(alpha5*V5/sigma5/np.sqrt(2)))
            G6 = (np.exp(-0.5 * (V6/sigma6)**2)) * (1+jax.scipy.special.erf(alpha6*V6/sigma6/np.sqrt(2)))
            G7 = (np.exp(-0.5 * (V7/sigma7)**2)) * (1+jax.scipy.special.erf(alpha7*V7/sigma7/np.sqrt(2)))
            G8 = (np.exp(-0.5 * (V8/sigma8)**2)) * (1+jax.scipy.special.erf(alpha8*V8/sigma8/np.sqrt(2)))
            if absorption:
                D = (np.exp(-A1 * G1/np.max(G1)) * 
                    np.exp(-A2 * G2/np.max(G2)) * 
                    np.exp(-A3 * G3/np.max(G3)) * 
                    np.exp(-A4 * G4/np.max(G4)) * 
                    np.exp(-A5 * G5/np.max(G5)) * 
                    np.exp(-A6 * G6/np.max(G6)) * 
                    np.exp(-A7 * G7/np.max(G7)) * 
                    np.exp(-A8 * G8/np.max(G8)) * 
                    poly)
            else:
                D = (A1 *G1/np.max(G1) + 
                    A2 * G2/np.max(G2) + 
                    A3 * G3/np.max(G3) + 
                    A4 * G4/np.max(G4) + 
                    A5 * G5/np.max(G5) + 
                    A6 * G6/np.max(G6) + 
                    A7 * G7/np.max(G7) + 
                    A8 * G8/np.max(G8) + 
                    poly)
            return D.mean(axis=1)
#The model for 1 line. ALSO CHANGE IN THE NUMPY CODE BELOW. THIS ONE IS THE NON-JITTED VERSION.
def gaussian_skewed_abs(x_super, A=0.0, x0=0.0,sigma=1.0,alpha=0,c0=1.0,c1=0.0,c2=0.0):
    """x goes in units of wavelength. Sigma goes in km/s. Absorption-line version of a Gaussian.
    A is the line center optical depth."""
    import numpy as np
    import jax.scipy.special
    c = 299792.458 #km/s
    X = (x_super-x0)
    V = X*c/x0
    poly = c0 + c1*V +c2*V*V
    G = (np.exp(-0.5 * (V/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V/sigma/np.sqrt(2)))
    D = np.exp(-A * G/np.max(G)) * poly
    return D.mean(axis=1)

#The model for 1 line. ALSO CHANGE IN THE NUMPY CODE BELOW. THIS ONE IS THE NON-JITTED VERSION.
def gaussian_skewed_triple(x_super, A1=0, x01=0.0,sigma1=1.0,alpha1=0,c0=0.0,c1=0.0,c2=0.0,A2=0.0,x02=0.0,sigma2=0.0,alpha2=0.0,A3=0.0,x03=0.0,sigma3=0.0,alpha3=0.0):
    """x goes in units of wavelength. Sigma goes in km/s. Allows for up to 3 components to be set."""
    import numpy as np
    import jax.scipy.special
    c = 299792.458 #km/s
    X1 = (x_super-x01)
    X2 = (x_super-x02)
    X3 = (x_super-x03)

    V1 = X1*c/x01
    V2 = X2*c/x02
    V3 = X3*c/x03

    G1 = (np.exp(-0.5 * (V1/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V1/sigma1/np.sqrt(2)))
    G2 = (np.exp(-0.5 * (V2/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V2/sigma2/np.sqrt(2)))
    G3 = (np.exp(-0.5 * (V3/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V3/sigma3/np.sqrt(2)))

    D = A1 * G1/np.max(G1) + A2 * G2/np.max(G2) + A3 * G3/np.max(G3) + c0 + c1*V1 +c2*V1*V1
    return D.mean(axis=1)

#The model for 1 line. ALSO CHANGE IN THE NUMPY CODE BELOW. THIS ONE IS THE NON-JITTED VERSION.
def gaussian_skewed_abs_triple(x_super, A1=0.0, x01=0.0,sigma1=1.0,alpha1=0,c0=1.0,c1=0.0,c2=0.0,A2=0.0,x02=0.0,sigma2=0.0,alpha2=0.0,A3=0.0,x03=0.0,sigma3=0.0,alpha3=0.0):
    """x goes in units of wavelength. Sigma goes in km/s. Absorption-line version of a Gaussian.
    A is the line center optical depth. Allows for up to 3 components to be set."""
    import numpy as np
    import jax.scipy.special
    c = 299792.458 #km/s
    X1 = (x_super-x01)
    X2 = (x_super-x02)
    X3 = (x_super-x03)
    V1 = X1*c/x01
    V2 = X2*c/x02
    V3 = X3*c/x03
    poly = c0 + c1*V1 +c2*V1*V1

    G1 = (np.exp(-0.5 * (V1/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V1/sigma1/np.sqrt(2)))
    G2 = (np.exp(-0.5 * (V2/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V2/sigma2/np.sqrt(2)))
    G3 = (np.exp(-0.5 * (V3/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V3/sigma3/np.sqrt(2)))

    D = np.exp(-A1 * G1/np.max(G1)) * np.exp(-A2 * G2/np.max(G2)) * np.exp(-A3 * G3/np.max(G3)) * poly
    return D.mean(axis=1)


def fit_lines(x,y,yerr,bounds,cpu_cores=4,oversample=10,progress_bar=True,nwarmup=200,nsamples=300,plot=True,absorption=True):
    """
    This function fits a (potentially skewed) Gaussian line profile to a chunk of spectrum.
    Multiple chunks of spectrum (currently up to 2) can be passed, and then this will fit those
    as a doublet, where the second component of the doublet as some separation from the first component,
    a line depth expressed as a ratio of the first componenent, and the same sigma-width and skew. 
    Each chunk also has its own polynomial-continuum up to degree 2 (cubic).

    The user should supply lower and upper bounds for uniform priors of all model parameters in the correc order.

    Depending on whether 1 or 2 components are set (by passing 1 or 2 chunks), these parameters are:

    beta,A,x0,sigma,alpha,c0,c1,c2. That's an array of len 8.
    beta,A,x0,sigma,alpha,c0,c1,c2,R,dx1,d0,d1,d2. Len 13.

    beta is uniform scaling of the noise. 
    A is the amplitude of the first component.
    x0 is the center wavelength of the first component.
    sigma is the Gaussian sigma width.
    alpha is the skew. Note that for small alpha (<2), sigma and alpha like to be degenerate, so you might consider switching this off.
    c0,c1 and c2 are the polynomial parameters (offset, linear, cubic).
    
    In case of two terms, the list goes on:
    R is the ratio between the two components. For R=2, the second line is half the strength of that of the first.
    dx1 is the separation (in units of wavelength) between the first and the second component.
    d0,d1 and d2 are again polynomial parameters for the second chunk.

    If you ever want to fit a triplet, you'd have to add a triplet-function with the additional free parameters. 
    If you want to use more polynomial parameters, you'll need to add those to the existing functions, and carefully
    restructure the reading-in of these parameters in the code below.


    Note that the fitting routine oversamples the line-function by a twice a factor (default is 10, that's 20x samples per original grid-point).

    

       Parameters
    ----------
    x : array-like
        1D array of the input wavelength range, matching the width of y and yerr, or a list of up to 2 of these.

    x : array-like
        1D array of the input wavelength range, matching the width of y and yerr, or a list of up to 2 of these.
    spec : array-like
        2D array of spectra time-series. 
        Its width matches the length of wl. 

    reject regions : list, tuple, optional
        An iterable of two-element iterables that contains the minimum and maximum wavelengths  
       to be ignored when fitting the continuum. Used to reject line regions.

    deg : int, optional
        The degree of the polynomial fit.

    plot: bool, optional
        Plotting for diagnostic check.


    Returns
    -------
    residuals : array
        The continuum-normalized spectrum.



    Examples
    --------
    >>> #Create a mock dataset with outliers in the negative direction.
    >>> import numpy.random as random
    >>> import tayph.functions as fun
    >>> import numpy as np
    >>> wl = np.arange(400,401,0.002)
    >>> S = random.normal(loc=1,scale=0.04,size=(400,len(wl)))
    >>> S+=fun.gaussian(wl,*[-0.2,400.5,0.1,-200,0.5])
    >>> S_norm = normslice(wl,S,reject_regions=[[400.2,400.25],[400.7,400.75]],deg=1,plot=False)   
    """

    import numpyro
    from numpyro.infer import MCMC, NUTS, Predictive
    from numpyro import distributions as dist
    # Set the number of cores on your machine for parallelism:
    numpyro.set_host_device_count(cpu_cores)
    import matplotlib.pyplot as plt
    import numpy as np
    from jax import config
    config.update("jax_enable_x64", True)
    import jax
    import jax.scipy.special
    from jax import jit, numpy as jnp
    from jax.random import PRNGKey, split
    import arviz
    from corner import corner
    from scipy.stats.distributions import norm


    #Current limitations
    max_components = 8 #max 8 line components per slice.
    degmax = 3 #max 3rd degree (cubic) continuum poly.
    maxterms = 2 #Max number of wl slices.

    try:
        void = len(x[0]) #If this fails then x[0] is a number which means thet x is passed as an array, not as a list of arrays.
        nterms = len(x) #If it doesn't fail, then x is passed as a list of arrays.
    except:
        nterms = 1 #If it failed there is only one component to fit.
        x=[x]#Force it into that list afterall.
        y=[y]
        yerr=[yerr]

    if nterms > maxterms:
        raise Exception(f'Only {maxterms} terms can be fit simultaneously in the current implementation. '
                        'Add more instances of line_model_n to accomodate more simultaneous fits.')

    if len(x) != len(y) or len(x) != len(yerr):
        raise Exception('x, y and yerr do not have the same length.') 


    #Force input to be jnp arrays.
    for i in range(len(x)):
        x[i] = jnp.array(x[i])
        y[i] = jnp.array(y[i])
        yerr[i] = jnp.array(yerr[i])

    # if nterms==1:
    #     x_supers = [supersample(x_i,f=oversample) for x_i in x]#This is a list.
    # if nterms==2:
    #     x_super1 = supersample(x[0],f=oversample)
    #     x_super2 = supersample(x[1],f=oversample)
    #     x_supers = [x_super1,x_super2]
    if nterms == 1:
        x_supers = supersample(x[0],f=oversample)
    else:
        x_supers = [supersample(x_i,f=oversample) for x_i in x]#This is a list.
    X = jnp.concatenate(x)#This is for plotting later.
    Y = jnp.concatenate(y)#This passes into the numpyro model.
    YERR = jnp.concatenate(yerr)# This too.


    #OK. Cleared the basic input. Now comes the big one: The parameter dictionaries.
    #First we test consistency:
    for k in bounds.keys():
        if hasattr(bounds[k], "__len__") and type(bounds[k]) != str: #if its a list. 
            if len(bounds[k]) > 2:
                raise Exception('Each value in bounds should be an array with no more than 2 elements.')
            if len(bounds[k]) == 2 and (bounds[k][0] > bounds[k][1]):
                raise Exception('All lower bounds should be smaller than the upper bounds.') 
        else:#Then its a number
            bounds[k] = [bounds[k],bounds[k]] #Set lower and upper limit to be identical. That fixes them in the sampler.
        



    #We are now going to determine what state we are in, and what input is required.
    #The following are mandatory and optional no matter what state we are in.
    mandatory = ['A1','mu1','sigma1']
    # optional  = ['alpha1']
    # optional += [f'c{i}' for i in range(degmax+1)]

    if nterms > 1:
        mandatory+=['R1','dx1']
        # optional += [f'd{i}' for i in range(degmax+1)]
    if nterms > 2:#This cannot be active in the current implentation but this is how we'd do it if there's 3 sections.
        mandatory+=['Q1','dx2']
        # optional += [f'e{i}' for i in range(degmax+1)]


    #How many components are required?
    nc = 1 #default.
    for i in range(2,max_components+1):
        if   f'A{i}' in bounds.keys() or f'mu{i}' in bounds.keys() or f'sigma{i}' in bounds.keys() or f'alpha{i}' in bounds.keys() or f'R{i}' in bounds.keys():
            nc = i #instead.
    #Then add the required free parameters:
    for i in range(1,max_components):
        if nc > i:
            mandatory+=[f'A{i+1}',f'mu{i+1}',f'sigma{i+1}']
            # optional.append(f'alpha{i+1}')
            if nterms > 1:
                mandatory.append(f'R{i+1}')
            if nterms > 2:
                mandatory.append(f'Q{i+1}')#Not used until 3 sections are supported.



    #Now we have a complete array of mandatory parameters.
    for k in mandatory:#Check that all mandatory parameters are present.
        if k not in bounds.keys():
            raise Exception(f'Given your input in the param dict, {k} has become a mandatory parameter. You need to set all of the following: {mandatory}.')


    #Now we determine what all possible free parameters could be, to test for reverse-consistency (junk in the param dict) and to set defaults.
    all_possible_params = ['beta']
    defaults = [1.0]
    for i in range(1,max_components+1):
        all_possible_params+=[f'A{i}',f'mu{i}',f'sigma{i}',f'alpha{i}',f'R{i}',f'Q{i}']
        defaults += [0.0,np.mean(bounds['mu1']),1.0,0.0,1.0,1.0]#Setting defaults like this is mostly to prevent div-0 errors anywhere.
    for i in range(0,degmax+1):
        all_possible_params+=[f'c{i}',f'd{i}',f'e{i}']
        if i == 0: 
            defaults += [1.0,1.0,1.0]#The default continuum is flat at 1.0, assuming a normalised input spectrum.
        else:
            defaults += [0.0,0.0,0.0]#but no other poly parameters. If you add more slices, these defaults also need to be set.
    all_possible_params+=['dx1','dx2']#Q, e{i} and dy1 are not used until a third slice is supported.
    defaults+=[0.0,0.0]

    #Check that the right number of defaults was set.
    if len(all_possible_params) != len(defaults):
        raise Exception(f"Lengths of defaults and possible parameters array don't match "
                        f"({len(all_possible_params)} vs {len(defaults)}). Check?")


    #To avoid errors, I'm being  strict in the passing of parameters in the dict. No junk allowed.
    for k in bounds.keys():
        if k not in all_possible_params:
            raise Exception(f'You are passing a keyword {k} that is not recognised. Please check.')

    #I'm putting the defaults in a dict for later look-up:
    def_dict = {}
    for i in range(len(all_possible_params)):
        def_dict[all_possible_params[i]] = defaults[i]



    #So now we have set all defaults as well. 
    #Finally we test that all inputs actually are legal:
        for k in bounds.keys():
            if k == 'beta' and (bounds[k][0] <= 0.0 or bounds[k][1] <= 0.0):
                raise Exception(f"beta may not be set to a non-positive value ({bounds[k][0]} , {bounds[k][1]}).")
            if k[0] == 'A' and (bounds[k][0] < 0.0 or bounds[k][1] < 0.0):
                if absorption == True:
                    raise Exception(f"A_n may not be set to a negative value ({bounds[k][0]} , {bounds[k][1]}).")
            if 'sigma' in k and (bounds[k][0] <= 0.0 or bounds[k][1] <= 0.0):
                raise Exception(f"sigma_n may not be set to a non-positive value ({bounds[k][0]} , {bounds[k][1]}).")
            if k[0] == 'R' and (bounds[k][0] <= 0.0 or bounds[k][1] <= 0.0):
                raise Exception(f"R_n may not be set to a non-positive value ({bounds[k][0]} , {bounds[k][1]}).")
            if k[0] == 'Q' and (bounds[k][0] <= 0.0 or bounds[k][1] <= 0.0):
                raise Exception(f"Q_n may not be set to a non-positive value ({bounds[k][0]} , {bounds[k][1]}).")
            




    #That should conclude all the inputs. Now we need to parse the inputs to define priors or fixed values, and the 
    #get-functions that go with them. Depending on what state we are in, these may not all be used, but I want to 
    #have a full logical collection of all parameters.
    #This used to be a big hard-coded mess with get_A1, get_A2, etc. all explicitly defined. I was able to do it 
    #programmatically by using the fact that functions are variables in python. You gotta love this stuff.
    import copy
    def spawn_get_function(name_local,bounds_local,def_dict_local):
        if name_local in bounds_local.keys():#If the parameter is user-defined then:
            if bounds_local[name_local][0] == bounds_local[name_local][1]:#... check if the bounds are the same.
                def get_param():#If so, the get_function fixes the value.
                    return(float(bounds_local[name_local][0]))
            else: 
                def get_param():#You could edit this here to add choice of prior function...!
                    return(numpyro.sample(name_local,dist.Uniform(low=bounds_local[name_local][0],high=bounds_local[name_local][1])))
        else:
            def get_param(): 
                return(float(def_dict_local[name_local]))
        return(copy.deepcopy(get_param))

    get_param_functions = {}#For each 
    for k in all_possible_params:
        get_param_functions[k] = spawn_get_function(k,bounds,def_dict)



    #Now we define our model functions. Hang on because this is many lines.
    #I define 2 flavours of function for whether or not we are fitting absorption lines or just gaussians,
    #another 2 for whether or not we have 1 or 2 slices.
    #another 2 for whether we have 1 component, or multiple (up to 8). The latter may be quite slow because of all the zeros
    #that get evaluated. So I may want to split that up into smaller numbers of components.
    if nc == 1 and nterms == 1 and absorption == False:
        @jit
        def line_model(x_super, A=0, mu=0.0,sigma=1.0,alpha=0,c0=0.0,c1=0.0,c2=0.0,c3=0.0):
            """x goes in units of wavelength. Sigma goes in km/s. x is an array of segments."""
            c = 299792.458 #km/s
            X = (x_super-mu)
            V = X*c/mu
            poly = c0 + c1*V + c2*V*V + c3*V*V*V
            G = (jnp.exp(-0.5 * (V/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V/sigma/jnp.sqrt(2)))
            D = A * G/jnp.max(G) + poly
            return D.mean(axis=1)
        

    elif nc == 1 and nterms == 1 and absorption == True:
        @jit
        def line_model(x_super, A=0, mu=0.0,sigma=1.0,alpha=0,c0=0.0,c1=0.0,c2=0.0,c3=0.0):
            """x goes in units of wavelength. Sigma goes in km/s."""
            c = 299792.458 #km/s
            X = (x_super-mu)
            V = X*c/mu
            poly = c0 + c1*V + c2*V*V + c3*V*V*V
            G = (jnp.exp(-0.5 * (V/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V/sigma/jnp.sqrt(2)))
            D = jnp.exp(-A * G/np.max(G)) * poly
            return D.mean(axis=1)
        

    elif nc > 1 and nterms == 1 and absorption == False:
        @jit
        def line_model(x_super, A1=0, mu1=0.0,sigma1=1.0,alpha1=0,c0=0.0,c1=0.0,c2=0.0,c3=0.0,
                        A2=0.0,mu2=0.0,sigma2=1.0,alpha2=0.0,
                        A3=0.0,mu3=0.0,sigma3=1.0,alpha3=0.0,
                        A4=0.0,mu4=0.0,sigma4=1.0,alpha4=0.0,
                        A5=0.0,mu5=0.0,sigma5=1.0,alpha5=0.0,                        
                        A6=0.0,mu6=0.0,sigma6=1.0,alpha6=0.0,
                        A7=0.0,mu7=0.0,sigma7=1.0,alpha7=0.0,
                        A8=0.0,mu8=0.0,sigma8=1.0,alpha8=0.0):
            """x goes in units of wavelength. Sigma goes in km/s. Allows for up to 8 components to be set."""
            c = 299792.458 #km/s
            X1 = x_super-mu1
            X2 = x_super-mu2
            X3 = x_super-mu3
            X4 = x_super-mu4
            X5 = x_super-mu5
            X6 = x_super-mu6
            X7 = x_super-mu7
            X8 = x_super-mu8
            V1 = X1*c/mu1
            V2 = X2*c/mu2
            V3 = X3*c/mu3
            V4 = X4*c/mu4
            V5 = X5*c/mu5
            V6 = X6*c/mu6
            V7 = X7*c/mu7
            V8 = X8*c/mu8
            G1 = (jnp.exp(-0.5 * (V1/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V1/sigma1/jnp.sqrt(2)))
            G2 = (jnp.exp(-0.5 * (V2/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V2/sigma2/jnp.sqrt(2)))
            G3 = (jnp.exp(-0.5 * (V3/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V3/sigma3/jnp.sqrt(2)))
            G4 = (jnp.exp(-0.5 * (V4/sigma4)**2)) * (1+jax.scipy.special.erf(alpha4*V4/sigma4/jnp.sqrt(2)))
            G5 = (jnp.exp(-0.5 * (V5/sigma5)**2)) * (1+jax.scipy.special.erf(alpha5*V5/sigma5/jnp.sqrt(2)))
            G6 = (jnp.exp(-0.5 * (V6/sigma6)**2)) * (1+jax.scipy.special.erf(alpha6*V6/sigma6/jnp.sqrt(2)))
            G7 = (jnp.exp(-0.5 * (V7/sigma7)**2)) * (1+jax.scipy.special.erf(alpha7*V7/sigma7/jnp.sqrt(2)))
            G8 = (jnp.exp(-0.5 * (V8/sigma8)**2)) * (1+jax.scipy.special.erf(alpha8*V8/sigma8/jnp.sqrt(2)))
            D = (A1 * G1/jnp.max(G1) + 
                 A2 * G2/jnp.max(G2) + 
                 A3 * G3/jnp.max(G3) + 
                 A4 * G4/jnp.max(G4) + 
                 A5 * G5/jnp.max(G5) + 
                 A6 * G6/jnp.max(G6) + 
                 A7 * G7/jnp.max(G7) + 
                 A8 * G8/jnp.max(G8) + 
                 c0 + c1*V1 + c2*V1**2 +c3*V1**3)
            return D.mean(axis=1)


    elif nc > 1 and nterms == 1 and absorption == True:
        @jit
        def line_model(x_super, A1=0, mu1=0.0,sigma1=1.0,alpha1=0,c0=0.0,c1=0.0,c2=0.0,c3=0.0,
                        A2=0.0,mu2=0.0,sigma2=1.0,alpha2=0.0,
                        A3=0.0,mu3=0.0,sigma3=1.0,alpha3=0.0,
                        A4=0.0,mu4=0.0,sigma4=1.0,alpha4=0.0,
                        A5=0.0,mu5=0.0,sigma5=1.0,alpha5=0.0,                        
                        A6=0.0,mu6=0.0,sigma6=1.0,alpha6=0.0,
                        A7=0.0,mu7=0.0,sigma7=1.0,alpha7=0.0,
                        A8=0.0,mu8=0.0,sigma8=1.0,alpha8=0.0):
            """x goes in units of wavelength. Sigma goes in km/s. Allows for up to 8 components to be set."""
            c = 299792.458 #km/s
            X1 = x_super-mu1
            X2 = x_super-mu2
            X3 = x_super-mu3
            X4 = x_super-mu4
            X5 = x_super-mu5
            X6 = x_super-mu6
            X7 = x_super-mu7
            X8 = x_super-mu8
            V1 = X1*c/mu1
            V2 = X2*c/mu2
            V3 = X3*c/mu3
            V4 = X4*c/mu4
            V5 = X5*c/mu5
            V6 = X6*c/mu6
            V7 = X7*c/mu7
            V8 = X8*c/mu8
            poly = c0 + c1*V1 + c2*V1**2 +c3*V1**3
            G1 = (jnp.exp(-0.5 * (V1/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V1/sigma1/jnp.sqrt(2)))
            G2 = (jnp.exp(-0.5 * (V2/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V2/sigma2/jnp.sqrt(2)))
            G3 = (jnp.exp(-0.5 * (V3/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V3/sigma3/jnp.sqrt(2)))
            G4 = (jnp.exp(-0.5 * (V4/sigma4)**2)) * (1+jax.scipy.special.erf(alpha4*V4/sigma4/jnp.sqrt(2)))
            G5 = (jnp.exp(-0.5 * (V5/sigma5)**2)) * (1+jax.scipy.special.erf(alpha5*V5/sigma5/jnp.sqrt(2)))
            G6 = (jnp.exp(-0.5 * (V6/sigma6)**2)) * (1+jax.scipy.special.erf(alpha6*V6/sigma6/jnp.sqrt(2)))
            G7 = (jnp.exp(-0.5 * (V7/sigma7)**2)) * (1+jax.scipy.special.erf(alpha7*V7/sigma7/jnp.sqrt(2)))
            G8 = (jnp.exp(-0.5 * (V8/sigma8)**2)) * (1+jax.scipy.special.erf(alpha8*V8/sigma8/jnp.sqrt(2)))
            D = (jnp.exp(-A1 * G1/jnp.max(G1)) * 
                 jnp.exp(-A2 * G2/jnp.max(G2)) * 
                 jnp.exp(-A3 * G3/jnp.max(G3)) * 
                 jnp.exp(-A4 * G4/jnp.max(G4)) * 
                 jnp.exp(-A5 * G5/jnp.max(G5)) * 
                 jnp.exp(-A6 * G6/jnp.max(G6)) * 
                 jnp.exp(-A7 * G7/jnp.max(G7)) * 
                 jnp.exp(-A8 * G8/jnp.max(G8)) * 
                 poly)
            return D.mean(axis=1)
    #We have now finished defining all cases where the number of slices is only one.


    elif nc == 1 and nterms == 2 and absorption == False:
        @jit
        def line_model(x_super, A=0, mu=0.0,sigma=1.0,alpha=0,c0=1.0,c1=0.0,c2=0.0,c3=0.0,R=0, dx1=0.0,d0=0.0,d1=0.0,d2=0.0,d3=0.0):
            """x goes in units of wavelength. Sigma goes in km/s.
            Alpha and sigma are the same for both components.
            This is an absorption line model, so it is multiplicative."""
            c = 299792.458 #km/s
            X1 = (x_super[0]-mu)
            X2 = (x_super[1]-(mu+dx1))
            V1 = X1*c / mu
            V2 = X2*c / (mu+dx1)
            poly1 = c0 + c1*V1 + c2*V1**2 + c3*V1**3
            poly2 = d0 + d1*V2 + d2*V2**2 + d3*V2**3
            G1 = (jnp.exp(-0.5 * (V1/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V1/sigma/jnp.sqrt(2)))
            G2 = (jnp.exp(-0.5 * (V2/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V2/sigma/jnp.sqrt(2)))
            D1 = A   * G1/jnp.max(G1) + poly1
            D2 = A/R * G2/jnp.max(G2) + poly2
            D = jnp.concatenate([D1,D2])
            return D.mean(axis=1)
        

    elif nc == 1 and nterms == 2 and absorption == True:
        @jit
        def line_model(x_super, A=0, mu=0.0,sigma=1.0,alpha=0,c0=1.0,c1=0.0,c2=0.0,c3=0.0,R=0, dx1=0.0,d0=0.0,d1=0.0,d2=0.0,d3=0.0):
            """x goes in units of wavelength. Sigma goes in km/s.
            Alpha and sigma are the same for both components.
            This is an absorption line model, so it is multiplicative."""
            c = 299792.458 #km/s
            X1 = x_super[0]-mu
            X2 = x_super[1]-(mu+dx1)
            V1 = X1*c / mu
            V2 = X2*c / (mu+dx1)
            poly1 = c0 + c1*V1 + c2*V1*V1 + c3*V1*V1*V1
            poly2 = d0 + d1*V2 + d2*V2**2 + d3*V2**3
            G1 = (jnp.exp(-0.5 * (V1/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V1/sigma/jnp.sqrt(2)))
            G2 = (jnp.exp(-0.5 * (V2/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V2/sigma/jnp.sqrt(2)))
            D1 = jnp.exp(-A   * G1/jnp.max(G1)) * poly1
            D2 = jnp.exp(-A/R * G2/jnp.max(G2)) * poly2
            D = jnp.concatenate([D1,D2])
            return D.mean(axis=1)
        

    elif nc > 1 and nterms == 2 and absorption == False:
        @jit
        def line_model(x_super,dx1=0.0,c0=0.0,c1=0.0,c2=0.0,c3=0.0,d0=0.0,d1=0.0,d2=0.0,d3=0.0,
                        A1=0.0,mu1=0.0,sigma1=1.0,alpha1=0.0,R1=1.0,
                        A2=0.0,mu2=0.0,sigma2=1.0,alpha2=0.0,R2=1.0,
                        A3=0.0,mu3=0.0,sigma3=1.0,alpha3=0.0,R3=1.0,
                        A4=0.0,mu4=0.0,sigma4=1.0,alpha4=0.0,R4=1.0,
                        A5=0.0,mu5=0.0,sigma5=1.0,alpha5=0.0,R5=1.0,                        
                        A6=0.0,mu6=0.0,sigma6=1.0,alpha6=0.0,R6=1.0,
                        A7=0.0,mu7=0.0,sigma7=1.0,alpha7=0.0,R7=1.0,
                        A8=0.0,mu8=0.0,sigma8=1.0,alpha8=0.0,R8=1.0):
            """x goes in units of wavelength. Sigma goes in km/s. Allows for up to 8 components to be set."""
            c = 299792.458 #km/s
            X11 = x_super[0]-mu1
            X12 = x_super[0]-mu2
            X13 = x_super[0]-mu3
            X14 = x_super[0]-mu4
            X15 = x_super[0]-mu5
            X16 = x_super[0]-mu6
            X17 = x_super[0]-mu7
            X18 = x_super[0]-mu8
            X21 = x_super[1]-(mu1+dx1)
            X22 = x_super[1]-(mu2+dx1)
            X23 = x_super[1]-(mu3+dx1)
            X24 = x_super[1]-(mu4+dx1)
            X25 = x_super[1]-(mu5+dx1)
            X26 = x_super[1]-(mu6+dx1)
            X27 = x_super[1]-(mu7+dx1)
            X28 = x_super[1]-(mu8+dx1)
            V11 = X11*c/mu1
            V12 = X12*c/mu2
            V13 = X13*c/mu3
            V14 = X14*c/mu4
            V15 = X15*c/mu5
            V16 = X16*c/mu6
            V17 = X17*c/mu7
            V18 = X18*c/mu8
            V21 = X21*c/(mu1+dx1)
            V22 = X22*c/(mu2+dx1)
            V23 = X23*c/(mu3+dx1)
            V24 = X24*c/(mu4+dx1)
            V25 = X25*c/(mu5+dx1)
            V26 = X26*c/(mu6+dx1)
            V27 = X27*c/(mu7+dx1)
            V28 = X28*c/(mu8+dx1)
            poly1 = c0 + c1*V11 + c2*V11**2 +c3*V11**3
            poly2 = d0 + d1*V21 + d2*V21**2 +d3*V21**3
            G11 = (jnp.exp(-0.5 * (V11/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V11/sigma1/jnp.sqrt(2)))
            G12 = (jnp.exp(-0.5 * (V12/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V12/sigma2/jnp.sqrt(2)))
            G13 = (jnp.exp(-0.5 * (V13/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V13/sigma3/jnp.sqrt(2)))
            G14 = (jnp.exp(-0.5 * (V14/sigma4)**2)) * (1+jax.scipy.special.erf(alpha4*V14/sigma4/jnp.sqrt(2)))
            G15 = (jnp.exp(-0.5 * (V14/sigma5)**2)) * (1+jax.scipy.special.erf(alpha5*V15/sigma5/jnp.sqrt(2)))
            G16 = (jnp.exp(-0.5 * (V14/sigma6)**2)) * (1+jax.scipy.special.erf(alpha6*V16/sigma6/jnp.sqrt(2)))
            G17 = (jnp.exp(-0.5 * (V14/sigma7)**2)) * (1+jax.scipy.special.erf(alpha7*V17/sigma7/jnp.sqrt(2)))
            G18 = (jnp.exp(-0.5 * (V14/sigma8)**2)) * (1+jax.scipy.special.erf(alpha8*V18/sigma8/jnp.sqrt(2)))
            G21 = (jnp.exp(-0.5 * (V21/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V21/sigma1/jnp.sqrt(2)))
            G22 = (jnp.exp(-0.5 * (V22/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V22/sigma2/jnp.sqrt(2)))
            G23 = (jnp.exp(-0.5 * (V23/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V23/sigma3/jnp.sqrt(2)))
            G24 = (jnp.exp(-0.5 * (V24/sigma4)**2)) * (1+jax.scipy.special.erf(alpha4*V24/sigma4/jnp.sqrt(2)))
            G25 = (jnp.exp(-0.5 * (V25/sigma5)**2)) * (1+jax.scipy.special.erf(alpha5*V25/sigma5/jnp.sqrt(2)))
            G26 = (jnp.exp(-0.5 * (V26/sigma6)**2)) * (1+jax.scipy.special.erf(alpha6*V26/sigma6/jnp.sqrt(2)))
            G27 = (jnp.exp(-0.5 * (V27/sigma7)**2)) * (1+jax.scipy.special.erf(alpha7*V27/sigma7/jnp.sqrt(2)))
            G28 = (jnp.exp(-0.5 * (V28/sigma8)**2)) * (1+jax.scipy.special.erf(alpha8*V28/sigma8/jnp.sqrt(2)))
            D1 =(A1 * G11/jnp.max(G11) + 
                 A2 * G12/jnp.max(G12) + 
                 A3 * G13/jnp.max(G13) + 
                 A4 * G14/jnp.max(G14) + 
                 A5 * G15/jnp.max(G15) + 
                 A6 * G16/jnp.max(G16) + 
                 A7 * G17/jnp.max(G17) + 
                 A8 * G18/jnp.max(G18) + 
                 poly1)
            D2 =(A1/R1 * G21/jnp.max(G21) + 
                 A2/R2 * G22/jnp.max(G22) + 
                 A3/R3 * G23/jnp.max(G23) + 
                 A4/R4 * G24/jnp.max(G24) + 
                 A5/R5 * G25/jnp.max(G25) + 
                 A6/R6 * G26/jnp.max(G26) + 
                 A7/R7 * G27/jnp.max(G27) + 
                 A8/R8 * G28/jnp.max(G28) + 
                 poly2)
            D = jnp.concatenate([D1,D2])
            return D.mean(axis=1)
        
    elif nc > 1 and nterms == 2 and absorption == True:
        @jit
        def line_model(x_super,dx1=0.0,c0=0.0,c1=0.0,c2=0.0,c3=0.0,d0=0.0,d1=0.0,d2=0.0,d3=0.0,
                        A1=0.0,mu1=0.0,sigma1=1.0,alpha1=0.0,R1=1.0,
                        A2=0.0,mu2=0.0,sigma2=1.0,alpha2=0.0,R2=1.0,
                        A3=0.0,mu3=0.0,sigma3=1.0,alpha3=0.0,R3=1.0,
                        A4=0.0,mu4=0.0,sigma4=1.0,alpha4=0.0,R4=1.0,
                        A5=0.0,mu5=0.0,sigma5=1.0,alpha5=0.0,R5=1.0,                        
                        A6=0.0,mu6=0.0,sigma6=1.0,alpha6=0.0,R6=1.0,
                        A7=0.0,mu7=0.0,sigma7=1.0,alpha7=0.0,R7=1.0,
                        A8=0.0,mu8=0.0,sigma8=1.0,alpha8=0.0,R8=1.0):
            """x goes in units of wavelength. Sigma goes in km/s. Allows for up to 8 components to be set."""
            c = 299792.458 #km/s
            X11 = x_super[0]-mu1
            X12 = x_super[0]-mu2
            X13 = x_super[0]-mu3
            X14 = x_super[0]-mu4
            X15 = x_super[0]-mu5
            X16 = x_super[0]-mu6
            X17 = x_super[0]-mu7
            X18 = x_super[0]-mu8
            X21 = x_super[1]-(mu1+dx1)
            X22 = x_super[1]-(mu2+dx1)
            X23 = x_super[1]-(mu3+dx1)
            X24 = x_super[1]-(mu4+dx1)
            X25 = x_super[1]-(mu5+dx1)
            X26 = x_super[1]-(mu6+dx1)
            X27 = x_super[1]-(mu7+dx1)
            X28 = x_super[1]-(mu8+dx1)
            V11 = X11*c/mu1
            V12 = X12*c/mu2
            V13 = X13*c/mu3
            V14 = X14*c/mu4
            V15 = X15*c/mu5
            V16 = X16*c/mu6
            V17 = X17*c/mu7
            V18 = X18*c/mu8
            V21 = X21*c/(mu1+dx1)
            V22 = X22*c/(mu2+dx1)
            V23 = X23*c/(mu3+dx1)
            V24 = X24*c/(mu4+dx1)
            V25 = X25*c/(mu5+dx1)
            V26 = X26*c/(mu6+dx1)
            V27 = X27*c/(mu7+dx1)
            V28 = X28*c/(mu8+dx1)
            poly1 = c0 + c1*V11 + c2*V11**2 +c3*V11**3
            poly2 = d0 + d1*V21 + d2*V21**2 +d3*V21**3
            G11 = (jnp.exp(-0.5 * (V11/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V11/sigma1/jnp.sqrt(2)))
            G12 = (jnp.exp(-0.5 * (V12/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V12/sigma2/jnp.sqrt(2)))
            G13 = (jnp.exp(-0.5 * (V13/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V13/sigma3/jnp.sqrt(2)))
            G14 = (jnp.exp(-0.5 * (V14/sigma4)**2)) * (1+jax.scipy.special.erf(alpha4*V14/sigma4/jnp.sqrt(2)))
            G15 = (jnp.exp(-0.5 * (V15/sigma5)**2)) * (1+jax.scipy.special.erf(alpha5*V15/sigma5/jnp.sqrt(2)))
            G16 = (jnp.exp(-0.5 * (V16/sigma6)**2)) * (1+jax.scipy.special.erf(alpha6*V16/sigma6/jnp.sqrt(2)))
            G17 = (jnp.exp(-0.5 * (V17/sigma7)**2)) * (1+jax.scipy.special.erf(alpha7*V17/sigma7/jnp.sqrt(2)))
            G18 = (jnp.exp(-0.5 * (V18/sigma8)**2)) * (1+jax.scipy.special.erf(alpha8*V18/sigma8/jnp.sqrt(2)))
            G21 = (jnp.exp(-0.5 * (V21/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V21/sigma1/jnp.sqrt(2)))
            G22 = (jnp.exp(-0.5 * (V22/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V22/sigma2/jnp.sqrt(2)))
            G23 = (jnp.exp(-0.5 * (V23/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V23/sigma3/jnp.sqrt(2)))
            G24 = (jnp.exp(-0.5 * (V24/sigma4)**2)) * (1+jax.scipy.special.erf(alpha4*V24/sigma4/jnp.sqrt(2)))
            G25 = (jnp.exp(-0.5 * (V25/sigma5)**2)) * (1+jax.scipy.special.erf(alpha5*V25/sigma5/jnp.sqrt(2)))
            G26 = (jnp.exp(-0.5 * (V26/sigma6)**2)) * (1+jax.scipy.special.erf(alpha6*V26/sigma6/jnp.sqrt(2)))
            G27 = (jnp.exp(-0.5 * (V27/sigma7)**2)) * (1+jax.scipy.special.erf(alpha7*V27/sigma7/jnp.sqrt(2)))
            G28 = (jnp.exp(-0.5 * (V28/sigma8)**2)) * (1+jax.scipy.special.erf(alpha8*V28/sigma8/jnp.sqrt(2)))
            D1 =(jnp.exp(-A1 * G11/jnp.max(G11)) * 
                 jnp.exp(-A2 * G12/jnp.max(G12)) * 
                 jnp.exp(-A3 * G13/jnp.max(G13)) * 
                 jnp.exp(-A4 * G14/jnp.max(G14)) * 
                 jnp.exp(-A5 * G15/jnp.max(G15)) * 
                 jnp.exp(-A6 * G16/jnp.max(G16)) * 
                 jnp.exp(-A7 * G17/jnp.max(G17)) * 
                 jnp.exp(-A8 * G18/jnp.max(G18)) * 
                 poly1)
            D2 =(jnp.exp(-A1/R1 * G21/jnp.max(G21)) * 
                 jnp.exp(-A2/R2 * G22/jnp.max(G22)) * 
                 jnp.exp(-A3/R3 * G23/jnp.max(G23)) * 
                 jnp.exp(-A4/R4 * G24/jnp.max(G24)) * 
                 jnp.exp(-A5/R5 * G25/jnp.max(G25)) * 
                 jnp.exp(-A6/R6 * G26/jnp.max(G26)) * 
                 jnp.exp(-A7/R7 * G27/jnp.max(G27)) * 
                 jnp.exp(-A8/R8 * G28/jnp.max(G28)) * 
                 poly2)
            D = jnp.concatenate([D1,D2])
            return D.mean(axis=1)
    else:
        raise Exception(f'No model function defined for your case (nc={nc},nterms={nterms},abs={absorption})')


    if nc == 1 and nterms == 1:
        def numpyro_model(predict=False):
            model_spectrum = line_model(x_supers, A=get_param_functions['A1'](), 
                                        mu=get_param_functions['mu1'](),
                                        sigma=get_param_functions['sigma1'](),
                                        alpha=get_param_functions['alpha1'](),
                                        c0=get_param_functions['c0'](),
                                        c1=get_param_functions['c1'](),
                                        c2=get_param_functions['c2'](),
                                        c3=get_param_functions['c3']())
            if predict:
                numpyro.deterministic("model_spectrum", model_spectrum)
            numpyro.sample("obs", dist.Normal(loc=model_spectrum,scale=get_param_functions['beta']()*YERR), obs=Y)

    elif nc > 1 and nterms == 1:
        def numpyro_model(predict=False):
            model_spectrum = line_model(x_supers,
                                        c0=get_param_functions['c0'](),
                                        c1=get_param_functions['c1'](),
                                        c2=get_param_functions['c2'](),
                                        c3=get_param_functions['c3'](),
                        A1=get_param_functions['A1'](),mu1=get_param_functions['mu1'](),sigma1=get_param_functions['sigma1'](),alpha1=get_param_functions['alpha1'](),
                        A2=get_param_functions['A2'](),mu2=get_param_functions['mu2'](),sigma2=get_param_functions['sigma2'](),alpha2=get_param_functions['alpha2'](),
                        A3=get_param_functions['A3'](),mu3=get_param_functions['mu3'](),sigma3=get_param_functions['sigma3'](),alpha3=get_param_functions['alpha3'](),
                        A4=get_param_functions['A4'](),mu4=get_param_functions['mu4'](),sigma4=get_param_functions['sigma4'](),alpha4=get_param_functions['alpha4'](),
                        A5=get_param_functions['A5'](),mu5=get_param_functions['mu5'](),sigma5=get_param_functions['sigma5'](),alpha5=get_param_functions['alpha5'](),                        
                        A6=get_param_functions['A6'](),mu6=get_param_functions['mu6'](),sigma6=get_param_functions['sigma6'](),alpha6=get_param_functions['alpha6'](),
                        A7=get_param_functions['A7'](),mu7=get_param_functions['mu7'](),sigma7=get_param_functions['sigma7'](),alpha7=get_param_functions['alpha7'](),
                        A8=get_param_functions['A8'](),mu8=get_param_functions['mu8'](),sigma8=get_param_functions['sigma8'](),alpha8=get_param_functions['alpha8']())
            if predict:
                numpyro.deterministic("model_spectrum", model_spectrum)
            numpyro.sample("obs", dist.Normal(loc=model_spectrum,scale=get_param_functions['beta']()*YERR), obs=Y)

    elif nc ==1 and nterms == 2:
        def numpyro_model(predict=False):
            model_spectrum = line_model(x_supers, 
                                        A=get_param_functions['A1'](), 
                                        mu=get_param_functions['mu1'](),
                                        sigma=get_param_functions['sigma1'](),
                                        alpha=get_param_functions['alpha1'](),
                                        c0=get_param_functions['c0'](),
                                        c1=get_param_functions['c1'](),
                                        c2=get_param_functions['c2'](),
                                        c3=get_param_functions['c3'](),
                                        d0=get_param_functions['d0'](),
                                        d1=get_param_functions['d1'](),
                                        d2=get_param_functions['d2'](),
                                        d3=get_param_functions['d3'](),
                                        R=get_param_functions['R1'](), dx1=get_param_functions['dx1']())        
            if predict:
                numpyro.deterministic("model_spectrum", model_spectrum)
            numpyro.sample("obs", dist.Normal(loc=model_spectrum,scale=get_param_functions['beta']()*YERR), obs=Y)

    elif nc > 1 and nterms == 2:
        def numpyro_model(predict=False):
            model_spectrum = line_model(x_supers,
                                        c0=get_param_functions['c0'](),
                                        c1=get_param_functions['c1'](),
                                        c2=get_param_functions['c2'](),
                                        c3=get_param_functions['c3'](),
                                        d0=get_param_functions['d0'](),
                                        d1=get_param_functions['d1'](),
                                        d2=get_param_functions['d2'](),
                                        d3=get_param_functions['d3'](),
                                        dx1=get_param_functions['dx1'](),
                        A1=get_param_functions['A1'](),mu1=get_param_functions['mu1'](),sigma1=get_param_functions['sigma1'](),alpha1=get_param_functions['alpha1'](),R1=get_param_functions['R1'](),
                        A2=get_param_functions['A2'](),mu2=get_param_functions['mu2'](),sigma2=get_param_functions['sigma2'](),alpha2=get_param_functions['alpha2'](),R2=get_param_functions['R2'](),
                        A3=get_param_functions['A3'](),mu3=get_param_functions['mu3'](),sigma3=get_param_functions['sigma3'](),alpha3=get_param_functions['alpha3'](),R3=get_param_functions['R3'](),
                        A4=get_param_functions['A4'](),mu4=get_param_functions['mu4'](),sigma4=get_param_functions['sigma4'](),alpha4=get_param_functions['alpha4'](),R4=get_param_functions['R4'](),
                        A5=get_param_functions['A5'](),mu5=get_param_functions['mu5'](),sigma5=get_param_functions['sigma5'](),alpha5=get_param_functions['alpha5'](),R5=get_param_functions['R5'](),                        
                        A6=get_param_functions['A6'](),mu6=get_param_functions['mu6'](),sigma6=get_param_functions['sigma6'](),alpha6=get_param_functions['alpha6'](),R6=get_param_functions['R6'](),
                        A7=get_param_functions['A7'](),mu7=get_param_functions['mu7'](),sigma7=get_param_functions['sigma7'](),alpha7=get_param_functions['alpha7'](),R7=get_param_functions['R7'](),
                        A8=get_param_functions['A8'](),mu8=get_param_functions['mu8'](),sigma8=get_param_functions['sigma8'](),alpha8=get_param_functions['alpha8'](),R8=get_param_functions['R8']())
            if predict:
                numpyro.deterministic("model_spectrum", model_spectrum)
            numpyro.sample("obs", dist.Normal(loc=model_spectrum,scale=get_param_functions['beta']()*YERR), obs=Y)
    else:
        raise Exception(f'No numpyro model defined for your case (nc={nc},nterms={nterms},abs={absorption}).')
    


    rng_seed = 0
    rng_keys = split(PRNGKey(rng_seed), cpu_cores)
    sampler = NUTS(numpyro_model,dense_mass=True)

    mcmc = MCMC(sampler, 
                     num_warmup=nwarmup, 
                     num_samples=nsamples, 
                     num_chains=cpu_cores,progress_bar=progress_bar)
    mcmc.run(rng_keys)


    samples = mcmc.get_samples()
    posterior_predictive = Predictive(model=numpyro_model,posterior_samples=samples,return_sites=['model_spectrum'])
    pred=posterior_predictive(rng_key=PRNGKey(1),predict=True)



    if plot:
        result = arviz.from_numpyro(mcmc)
        corner(result,quiet=True)
        plt.savefig('cornerplot.png',dpi=250)
        plt.show()


        from scipy.stats.distributions import norm
        beta=np.median(samples['beta'])
        low_1, mid_1, high_1 = np.percentile(pred['model_spectrum'], 100 * norm.cdf([-1, 0,1]), axis=0)
        low_2, mid_2, high_2 = np.percentile(pred['model_spectrum'], 100 * norm.cdf([-2, 0,2]), axis=0)
        low_3, mid_3, high_3 = np.percentile(pred['model_spectrum'], 100 * norm.cdf([-3, 0,3]), axis=0)
        fig,ax = plt.subplots(1,len(x),figsize=(14,5))
        if len(x) == 1:
            ax=[ax,0]
        for i in range(len(x)):
            sel = (X>np.min(x[i]))&(X<np.max(x[i]))
            ax[i].plot(X[sel], mid_1[sel],label='Mean model',color='red')
            ax[i].plot(X[sel], Y[sel]-mid_1[sel],color='grey',label='Residual')
            ax[i].fill_between(X[sel], low_1[sel], high_1[sel], alpha=0.2, color='DodgerBlue',label='1,2,3 $\sigma$ posterior')
            ax[i].fill_between(X[sel], low_2[sel], high_2[sel], alpha=0.2, color='DodgerBlue')
            ax[i].fill_between(X[sel], low_3[sel], high_3[sel], alpha=0.2, color='DodgerBlue')
            ax[i].errorbar(X[sel],Y[sel],fmt='.',yerr=beta*YERR[sel],label='Data & scaled error',color='black')
            ax[i].set_xlabel('Pixel position')
            ax[i].set_ylabel('Flux')
        ax[0].legend() 
        plt.savefig('bestfit.png',dpi=250)
        plt.show()


    return(mcmc)



### TO DEMONSTRATE THAT THIS CODE WORKS.
def test(nc=1,nterms=1,absorption=True):
    import numpy.random
    import matplotlib.pyplot as plt
    import numpy as np
    noise = 0.02
    sigma = 15.0
    alpha = 0.0
    c0 = 1.0
    c1 = 0.0
    c2 = 0.0
    c3 = 0.0

    x1 = np.arange(579.80,581.2,0.01)
    x2 = np.arange(min(x1)+3,max(x1)+3,0.008)
    xs1 = supersample(x1)    
    xs2 = supersample(x2)
    yerr1 = np.zeros_like(x1)+noise
    yerr2 = np.zeros_like(x2)+noise

    if nc == 1:
        A1 = 0.8
        mu1 = 580.5
        A2 = 0.4
        mu2 = 583.5
        y1 = gaussian_skewed(xs1,A1=A1,mu1=mu1,sigma1=sigma,alpha1=alpha,c0=c0,c1=c1,c2=c2,c3=c3,absorption=absorption)+numpy.random.normal(scale=noise,size=len(x1))
        y2 = gaussian_skewed(xs2,A1=A2,mu1=mu2,sigma1=sigma,alpha1=alpha,c0=c0,c1=c1,c2=c2,c3=c3,absorption=absorption)+numpy.random.normal(scale=noise,size=len(x2))
        bounds = {}
        bounds['beta'] = [0.5,2.0]
        bounds['A1'] = [0,2]
        bounds['mu1'] = [mu1-0.1,mu1+0.1]
        bounds['sigma1'] = [5,30]
        bounds['alpha1'] = [-3,3]
        bounds['c0'] = [0.9,1.1]
        bounds['c1'] = [-1e-2,1e-2]
        bounds['c2'] = [-1e-4,1e-4]
        bounds['c3'] = [-1e-6,1e-6]
        if nterms == 2:
            bounds['d0'] = [0.9,1.1]
            bounds['d1'] = [-1e-2,1e-2]
            bounds['d2'] = [-1e-4,1e-4]
            bounds['d3'] = [-1e-6,1e-6]
            bounds['R1'] = [0.25,4]
            bounds['dx1'] = [3-0.05,3+0.05]


    if nc == 3:
        A11 = 0.8
        mu11 = 580.5
        A12 = 0.6
        mu12 = 580.7
        A13 = 0.4
        mu13 = 580.4
        A21 = 0.4
        mu21 = 583.5
        A22 = 0.3
        mu22 = 583.7
        A23 = 0.2
        mu23 = 583.4
        bounds = {}
        bounds['beta'] = [0.5,2.0]
        bounds['A1'] = [0,2]
        bounds['mu1'] = [mu11-0.05,mu11+0.05]
        bounds['sigma1'] = [5,30]
        bounds['A2'] = [0,2]
        bounds['mu2'] = [mu12-0.05,mu12+0.05]
        bounds['sigma2'] = [5,30]
        bounds['A3'] = [0,2]
        bounds['mu3'] = [mu13-0.05,mu13+0.05]
        bounds['sigma3'] = [5,30]
        bounds['c0'] = [0.9,1.1]
        bounds['c1'] = [-1e-2,1e-2]
        bounds['c2'] = [-1e-4,1e-4]
        bounds['c3'] = [-1e-6,1e-6]
        if nterms == 2:
            bounds['d0'] = [0.9,1.1]
            bounds['d1'] = [-1e-2,1e-2]
            bounds['d2'] = [-1e-4,1e-4]
            bounds['d3'] = [-1e-6,1e-6]
            bounds['R1'] = [0.25,4]
            bounds['R2'] = [0.25,4]
            bounds['R3'] = [0.25,4]
            bounds['dx1'] = [3-0.05,3+0.05]
        y1 = gaussian_skewed(xs1,A1=A11,mu1=mu11,sigma1=sigma,alpha1=alpha,A2=A12,mu2=mu12,sigma2=sigma,alpha2=alpha,A3=A13,mu3=mu13,sigma3=sigma,alpha3=alpha,c0=c0,c1=c1,c2=c2,c3=c3,absorption=absorption)+numpy.random.normal(scale=noise,size=len(x1))
        y2 = gaussian_skewed(xs2,A1=A21,mu1=mu21,sigma1=sigma,alpha1=alpha,A2=A22,mu2=mu22,sigma2=sigma,alpha2=alpha,A3=A23,mu3=mu23,sigma3=sigma,alpha3=alpha,c0=c0,c1=c1,c2=c2,c3=c3,absorption=absorption)+numpy.random.normal(scale=noise,size=len(x2))

    if nc == 6:
        A11 = 2.0
        mu11 = 580.5
        A12 = 1.0
        mu12 = 580.7
        A13 = 0.8
        mu13 = 580.42
        A14 = 0.8
        mu14 = 580.1
        A15 = 0.6
        mu15 = 580.9
        A16 = 0.4
        mu16 = 580.3

        A21 = A11/2
        mu21 = mu11+3
        A22 = A12/2
        mu22 = mu12+3
        A23 = A13/2
        mu23 = mu13+3
        A24 = A14/2
        mu24 = mu14+3
        A25 = A15/2
        mu25 = mu15+3
        A26 = A16/2
        mu26 = mu16+3
        bounds = {}
        bounds['beta'] = [0.5,2.0]
        bounds['A1'] = [0,6]
        bounds['mu1'] = [mu11-0.03,mu11+0.03]
        bounds['sigma1'] = [5,30]
        bounds['A2'] = [0,2]
        bounds['mu2'] = [mu12-0.03,mu12+0.03]
        bounds['sigma2'] = [5,25]
        bounds['A3'] = [0,2]
        bounds['mu3'] = [mu13-0.03,mu13+0.03]
        bounds['sigma3'] = [5,25]
        bounds['A4'] = [0,2]
        bounds['mu4'] = [mu14-0.03,mu14+0.03]
        bounds['sigma4'] = [5,25]
        bounds['A5'] = [0,2]
        bounds['mu5'] = [mu15-0.03,mu15+0.03]
        bounds['sigma5'] = [5,25]
        bounds['A6'] = [0,1]
        bounds['mu6'] = [mu16-0.03,mu16+0.03]
        bounds['sigma6'] = [5,25]
        bounds['c0'] = [0.97,1.03]
        bounds['c1'] = [-1e-3,1e-3]
        bounds['c2'] = [-2e-6,2e-6]
        bounds['c3'] = [-2e-8,2e-8]
        if nterms == 2:
            bounds['d0'] = [0.97,1.03]
            bounds['d1'] = [-1e-2,1e-2]
            bounds['d2'] = [-1e-4,1e-4]
            bounds['d3'] = [-1e-6,1e-6]
            bounds['R1'] = [1,3]
            bounds['R2'] = [1,3]
            bounds['R3'] = [1,3]
            bounds['R4'] = [1,3]
            bounds['R5'] = [1,3]
            bounds['R6'] = [1,3]
            bounds['dx1'] = [3-0.05,3+0.05]
        y1 = gaussian_skewed(xs1,A1=A11,mu1=mu11,sigma1=sigma,alpha1=alpha,A2=A12,mu2=mu12,sigma2=sigma,alpha2=alpha,A3=A13,mu3=mu13,sigma3=sigma,alpha3=alpha,
                             A4=A14,mu4=mu14,sigma4=sigma,alpha4=alpha,A5=A15,mu5=mu15,sigma5=sigma,A6=A16,mu6=mu16,sigma6=sigma,alpha6=alpha,
                             c0=c0,c1=c1,c2=c2,c3=c3,absorption=absorption)+numpy.random.normal(scale=noise,size=len(x1))
        y2 = gaussian_skewed(xs2,A1=A21,mu1=mu21,sigma1=sigma,alpha1=alpha,A2=A22,mu2=mu22,sigma2=sigma,alpha2=alpha,A3=A23,mu3=mu23,sigma3=sigma,alpha3=alpha,
                             A4=A24,mu4=mu24,sigma4=sigma,alpha4=alpha,A5=A25,mu5=mu25,sigma5=sigma,A6=A26,mu6=mu26,sigma6=sigma,alpha6=alpha,
                             c0=c0,c1=c1,c2=c2,c3=c3,absorption=absorption)+numpy.random.normal(scale=noise,size=len(x2))



    if nc == 8:
        A11 = 2.0
        mu11 = 580.5
        A12 = 1.0
        mu12 = 580.7
        A13 = 0.8
        mu13 = 580.42
        A14 = 0.8
        mu14 = 580.1
        A15 = 0.6
        mu15 = 580.9
        A16 = 0.4
        mu16 = 580.3
        A17 = 0.6
        mu17 = 580.8
        A18 = 0.4
        mu18 = 580.05


        A21 = A11/2
        mu21 = mu11+3
        A22 = A12/2
        mu22 = mu12+3
        A23 = A13/2
        mu23 = mu13+3
        A24 = A14/2
        mu24 = mu14+3
        A25 = A15/2
        mu25 = mu15+3
        A26 = A16/2
        mu26 = mu16+3
        A27 = A17/2
        mu27 = mu17+3
        A28 = A18/2
        mu28 = mu18+3
        bounds = {}
        bounds['beta'] = [0.5,2.0]
        bounds['A1'] = [0,6]
        bounds['mu1'] = [mu11-0.03,mu11+0.03]
        bounds['sigma1'] = [5,30]
        bounds['A2'] = [0,2]
        bounds['mu2'] = [mu12-0.03,mu12+0.03]
        bounds['sigma2'] = [5,25]
        bounds['A3'] = [0,2]
        bounds['mu3'] = [mu13-0.03,mu13+0.03]
        bounds['sigma3'] = [5,25]
        bounds['A4'] = [0,2]
        bounds['mu4'] = [mu14-0.03,mu14+0.03]
        bounds['sigma4'] = [5,25]
        bounds['A5'] = [0,2]
        bounds['mu5'] = [mu15-0.03,mu15+0.03]
        bounds['sigma5'] = [5,25]
        bounds['A6'] = [0,1]
        bounds['mu6'] = [mu16-0.03,mu16+0.03]
        bounds['sigma6'] = [5,25]
        bounds['A7'] = [0,2]
        bounds['mu7'] = [mu17-0.03,mu17+0.03]
        bounds['sigma7'] = [5,25]
        bounds['A8'] = [0,1]
        bounds['mu8'] = [mu18-0.03,mu18+0.03]
        bounds['sigma8'] = [5,25]
        bounds['c0'] = [0.97,1.03]
        bounds['c1'] = [-1e-3,1e-3]
        bounds['c2'] = [-2e-6,2e-6]
        bounds['c3'] = [-2e-8,2e-8]
        if nterms == 2:
            bounds['d0'] = [0.97,1.03]
            bounds['d1'] = [-1e-2,1e-2]
            bounds['d2'] = [-1e-4,1e-4]
            bounds['d3'] = [-1e-6,1e-6]
            bounds['R1'] = [1,3]
            bounds['R2'] = [1,3]
            bounds['R3'] = [1,3]
            bounds['R4'] = [1,3]
            bounds['R5'] = [1,3]
            bounds['R6'] = [1,3]
            bounds['R7'] = [1,3]
            bounds['R8'] = [1,3]
            bounds['dx1'] = [3-0.05,3+0.05]
        y1 = gaussian_skewed(xs1,A1=A11,mu1=mu11,sigma1=sigma,alpha1=alpha,A2=A12,mu2=mu12,sigma2=sigma,alpha2=alpha,A3=A13,mu3=mu13,sigma3=sigma,alpha3=alpha,
                             A4=A14,mu4=mu14,sigma4=sigma,alpha4=alpha,A5=A15,mu5=mu15,sigma5=sigma,A6=A16,mu6=mu16,sigma6=sigma,alpha6=alpha,
                             A7=A27,mu7=mu27,sigma7=sigma,alpha7=alpha,A8=A28,mu8=mu28,sigma8=sigma,alpha8=alpha,
                             c0=c0,c1=c1,c2=c2,c3=c3,absorption=absorption)+numpy.random.normal(scale=noise,size=len(x1))
        y2 = gaussian_skewed(xs2,A1=A21,mu1=mu21,sigma1=sigma,alpha1=alpha,A2=A22,mu2=mu22,sigma2=sigma,alpha2=alpha,A3=A23,mu3=mu23,sigma3=sigma,alpha3=alpha,
                             A4=A24,mu4=mu24,sigma4=sigma,alpha4=alpha,A5=A25,mu5=mu25,sigma5=sigma,A6=A26,mu6=mu26,sigma6=sigma,alpha6=alpha,
                             A7=A27,mu7=mu27,sigma7=sigma,alpha7=alpha,A8=A28,mu8=mu28,sigma8=sigma,alpha8=alpha,
                             c0=c0,c1=c1,c2=c2,c3=c3,absorption=absorption)+numpy.random.normal(scale=noise,size=len(x2))



    if nterms==1:
        plt.errorbar(x1,y1,yerr=yerr1,color='black')
        plt.title('Data to to fit') 
        plt.show()  
        fit_lines([x1],[y1],[yerr1],bounds,cpu_cores=3,oversample=5,progress_bar=True,nwarmup=800,nsamples=400)
    if nterms==2:
        fig,ax = plt.subplots(1,2,figsize=(14,5),sharey=True)

        ax[0].errorbar(x1,y1,yerr=yerr1,color='black')
        ax[1].errorbar(x2,y2,yerr=yerr2,color='black')
        ax[0].set_title('Data to to fit')
        ax[1].set_title('Data to fit')
        plt.show()
        fit_lines([x1,x2],[y1,y2],[yerr1,yerr2],bounds,cpu_cores=3,oversample=5,progress_bar=True,nwarmup=800,nsamples=400)
