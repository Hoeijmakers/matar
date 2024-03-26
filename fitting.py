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
def gaussian_skewed(x_super, A=0, x0=0.0,sigma=1.0,alpha=0,c0=0.0,c1=0.0,c2=0.0):
    """x goes in units of wavelength. Sigma goes in km/s."""
    import numpy as np
    import jax.scipy.special
    c = 299792.458 #km/s
    X = (x_super-x0)
    V = X*c/x0
    G = (np.exp(-0.5 * (V/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V/sigma/np.sqrt(2)))
    D = A * G/np.max(G) + c0 + c1*V +c2*V*V
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


def fit_lines(x,y,yerr,lower,upper,cpu_cores=4,oversample=10,progress_bar=True,nwarmup=200,nsamples=300,plot=True):
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


    try:
        void = len(x[0]) #If this fails then x[0] is a number which means thet x is passed as an array, not as a list of arrays.
        nterms = len(x) #If it doesn't fail, then x is passed as a list of arrays.
    except:
        nterms = 1 #If it failed there is only one component to fit.
        x=[x]#Force it into that list afterall.
        y=[y]
        yerr=[yerr]

    if nterms > 2:
        raise Exception('Only 2 terms can be fit simultaneously in the current implementation. Add more instances of line_model_n to accomodate more simultaneous fits.')

    if len(x) != len(y) or len(x) != len(yerr):
        raise Exception('x, y and yerr do not have the same length.') 


 
    #Force things to be jnp arrays.
    for i in range(len(x)):
        x[i] = jnp.array(x[i])
        y[i] = jnp.array(y[i])
        yerr[i] = jnp.array(yerr[i])

    
    if nterms==1:
        x_supers = [supersample(x[0],f=oversample)]
    if nterms==2:
        x_super1 = supersample(x[0],f=oversample)
        x_super2 = supersample(x[1],f=oversample)
        x_supers = [x_super1,x_super2]




    #OK. Cleared the basic input. Now comes the big one: The parameter dictionaries.
    #First we test consitency:
        

    for k in lower.keys():
        if k not in upper.keys():#Anything set only in the lower bound, is interpreted to be fixed to that value.
            upper[k]=lower[k]
        if lower[k] > upper[k]:
            raise Exception('All lower bounds should be smaller than the upper bounds.') 
        

    #We are now going to determine what state we are in, and what input is required.
    #First the number of components:
    if 'A8' in lower.keys() or 'mu8' in lower.keys() or 'sigma8' in lower.keys():
        nc = 8
    elif 'A7' in lower.keys() or 'mu7' in lower.keys() or 'sigma7' in lower.keys():
        nc = 7
    elif 'A6' in lower.keys() or 'mu6' in lower.keys() or 'sigma6' in lower.keys():
        nc = 6
    elif 'A5' in lower.keys() or 'mu5' in lower.keys() or 'sigma5' in lower.keys():
        nc = 5
    elif 'A4' in lower.keys() or 'mu4' in lower.keys() or 'sigma4' in lower.keys():
        nc = 4
    elif 'A3' in lower.keys() or 'mu3' in lower.keys() or 'sigma3' in lower.keys():
        nc = 3
    elif 'A2' in lower.keys() or 'mu2' in lower.keys() or 'sigma2' in lower.keys():
        nc = 2
    else:
        nc = 1

    #The following are mandatory and optional no matter what state we are in.
    mandatory = ['beta','A1','mu1','sigma1']
    optional  = ['alpha1','c0','c1','c2','c3']

    if nterms > 1:
        mandatory+=['R1','dx1']
        optional+=['d0','d1','d2','d3']
    if nterms > 2:#This cannot be active in the current implentation but this is how we'd do it if there's 3 sections.
        mandatory+=['Q1','dy1']
        optional+=['e0','e1','e2','e3']

    for i in range(1,8):
        if nc > i:
            mandatory+=[f'A{i+1}',f'mu{i+1}',f'sigma{i+1}']
            optional.append(f'alpha{i+1}')
            if nterms > 1:
                mandatory.append(f'R{i+1}')
            if nterms > 2:
                mandatory.append(f'Q{i+1}')

    #Now we have an array of mandatory and optional parameters.
    print(mandatory)
    print(optional)
    print(nc)

    import sys
    sys.exit()
    #The following set of free parameters is the minimum mandatory.
    if lower['beta'] == upper['beta']:
        def get_beta(): return(lower['beta'])
    else: 
        def get_beta(): return(numpyro.sample('beta',dist.Uniform(low=lower['beta'],high=upper['beta'])))
    if lower['A1']==upper['A1']: 
        def get_A1(): return(lower['A1'])
    else: 
        def get_A1(): return(numpyro.sample('A1',dist.Uniform(low=lower['A1'],high=upper['A1'])))       
    if lower['mu1']==upper['mu1']: 
        def get_mu1(): return(lower['mu1'])
    else: 
        def get_mu1(): return(numpyro.sample('mu1',dist.Uniform(low=lower['mu1'],high=upper['mu1'])))
    if lower['sigma1']==upper['sigma1']: 
        def get_sigma1(): return(lower['sigma1'])
    else: 
        def get_sigma1(): return(numpyro.sample('sigma1',dist.Uniform(low=lower['sigma1'],high=upper['sigma1'])))
    if lower['alpha1']==upper['alpha1']: 
        def get_alpha1(): return(lower['alpha1'])
    else: 
        def get_alpha1(): return(numpyro.sample('alpha1',dist.Uniform(low=lower['alpha1'],high=upper['alpha1'])))

    #The following are optional for the polynomial of the first section:
    if 'c0' in lower.keys():
        if lower['c0']==upper['c0']: 
            def get_c0(): return(lower['c0'])
        else: 
            def get_c0(): return(numpyro.sample('c0',dist.Uniform(low=lower['c0'],high=upper['c0'])))
    if 'c1' in lower.keys():
        if lower['c1']==upper['c1']: 
            def get_c1(): return(lower['c1'])
        else: 
            def get_c1(): return(numpyro.sample('c1',dist.Uniform(low=lower['c1'],high=upper['c1'])))
    if 'c2' in lower.keys():   
        if lower['c2']==upper['c2']: 
            def get_c2(): return(lower['c2'])
        else: 
            def get_c2(): return(numpyro.sample('c2',dist.Uniform(low=lower['c2'],high=upper['c2'])))
    if 'c3' in lower.keys():
        if lower['c3']==upper['c3']: 
            def get_c3(): return(lower['c3'])
        else: 
            def get_c3(): return(numpyro.sample('c3',dist.Uniform(low=lower['c3'],high=upper['c3'])))

    

    #If we are doing a second section, the first two are mandatory:
    if nterms > 1:
        if lower['R1']==upper['R1']: 
            def get_R1(): return(lower['R1'])
        else: 
            def get_R1(): return(numpyro.sample('R1',dist.Uniform(low=lower['R1'],high=upper['R1'])))
        if lower['dx1']==upper['dx1']: 
            def get_dx1(): return(lower['dx1'])
        else: 
            def get_dx1(): return(numpyro.sample('dx1',dist.Uniform(low=lower['dx1'],high=upper['dx1'])))

        #If we are doing a second section, these are optional:
        if 'd0' in lower.keys():
            if lower['d0']==upper['d0']: 
                def get_d0(): return(lower['d0'])
            else: 
                def get_d0(): return(numpyro.sample('d0',dist.Uniform(low=lower['d0'],high=upper['d0'])))
        if 'd1' in lower.keys():
            if lower['d1']==upper['d1']: 
                def get_d1(): return(lower['d1'])
            else: 
                def get_d1(): return(numpyro.sample('d1',dist.Uniform(low=lower['d1'],high=upper['d1'])))
        if 'd2' in lower.keys():
            if lower['d2']==upper['d2']: 
                def get_d2(): return(lower['d2'])
            else: 
                def get_d2():  return(numpyro.sample('d2',dist.Uniform(low=lower['d2'],high=upper['d2'])))
        if 'd3' in lower.keys():
            if lower['d3']==upper['d3']: 
                def get_d3(): return(lower['d3'])
            else: 
                def get_d3():  return(numpyro.sample('d3',dist.Uniform(low=lower['d3'],high=upper['d3'])))




    if nc > 1:
        if lower['A2']==upper['A2']: 
            def get_A2(): return(lower['A2'])
        else: 
            def get_A2(): return(numpyro.sample('A2',dist.Uniform(low=lower['A2'],high=upper['A2'])))       
        if lower['mu2']==upper['mu2']: 
            def get_mu2(): return(lower['mu2'])
        else: 
            def get_mu2(): return(numpyro.sample('mu2',dist.Uniform(low=lower['mu2'],high=upper['mu2'])))
        if lower['sigma2']==upper['sigma2']: 
            def get_sigma2(): return(lower['sigma2'])
        else: 
            def get_sigma2(): return(numpyro.sample('sigma2',dist.Uniform(low=lower['sigma2'],high=upper['sigma2'])))
        if lower['alpha2']==upper['alpha2']: 
            def get_alpha2(): return(lower['alpha2'])
        else: 
            def get_alpha2(): return(numpyro.sample('alpha2',dist.Uniform(low=lower['alpha2'],high=upper['alpha2']))) 
        if nterms > 1:
            if lower['R2']==upper['R2']: 
                def get_R2(): return(lower['R2'])
            else: 
                def get_R2(): return(numpyro.sample('R2',dist.Uniform(low=lower['R2'],high=upper['R2'])))
    if nc == 3:
        if lower['A2']==upper['A2']: 
            def get_A2(): return(lower['A2'])
        else: 
            def get_A2(): return(numpyro.sample('A2',dist.Uniform(low=lower['A2'],high=upper['A2'])))       
        if lower['mu2']==upper['mu2']: 
            def get_mu2(): return(lower['mu2'])
        else: 
            def get_mu2(): return(numpyro.sample('mu2',dist.Uniform(low=lower['mu2'],high=upper['mu2'])))
        if lower['sigma2']==upper['sigma2']: 
            def get_sigma2(): return(lower['sigma2'])
        else: 
            def get_sigma2(): return(numpyro.sample('sigma2',dist.Uniform(low=lower['sigma2'],high=upper['sigma2'])))
        if lower['alpha2']==upper['alpha2']: 
            def get_alpha2(): return(lower['alpha2'])
        else: 
            def get_alpha2(): return(numpyro.sample('alpha2',dist.Uniform(low=lower['alpha2'],high=upper['alpha2']))) 
        if nterms > 1:
            if lower['R2']==upper['R2']: 
                def get_R2(): return(lower['R2'])
            else: 
                def get_R2(): return(numpyro.sample('R2',dist.Uniform(low=lower['R2'],high=upper['R2'])))








    X = jnp.concatenate(x)#This is for plotting later.
    Y = jnp.concatenate(y)#This passes into the numpyro model.
    YERR = jnp.concatenate(yerr)
 
    @jit#The model for 1 line.
    def line_model_1(x_super, A=0, x0=0.0,sigma=1.0,alpha=0,c0=0.0,c1=0.0,c2=0.0):
        """x goes in units of wavelength. Sigma goes in km/s."""
        c = 299792.458 #km/s
        X = (x_super-x0)
        V = X*c/x0
        poly = c0 + c1*V +c2*V*V
        G = (jnp.exp(-0.5 * (V/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V/sigma/jnp.sqrt(2)))
        D = A * G/jnp.max(G) + poly
        return D.mean(axis=1)
    
    @jit#The model for 1 line in absorption.
    def line_model_1_abs(x_super, A=0, x0=0.0,sigma=1.0,alpha=0,c0=1.0,c1=0.0,c2=0.0):
        """x goes in units of wavelength. Sigma goes in km/s. Absorption-line version of a Gaussian.
        A is the line center optical depth."""
        import numpy as np
        import jax.scipy.special
        c = 299792.458 #km/s
        X = (x_super-x0)
        V = X*c/x0
        poly = c0 + c1*V +c2*V*V
        G = (jnp.exp(-0.5 * (V/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V/sigma/jnp.sqrt(2)))
        D = jnp.exp(-A * G/np.max(G)) * poly
        return D.mean(axis=1)

    @jit#The model for 2 lines.
    def line_model_2(x_super, A=0, x0=0.0,sigma=1.0,alpha=0,c0=1.0,c1=0.0,c2=0.0,R=0, dx1=0.0,d0=0.0,d1=0.0,d2=0.0):
        """x goes in units of wavelength. Sigma goes in km/s.
        Alpha and sigma are the same for both components.
        This is an absorption line model, so it is multiplicative."""
        c = 299792.458 #km/s
        X1 = (x_super[0]-x0)
        X2 = (x_super[1]-(x0+dx1))
        V1 = X1*c / x0
        V2 = X2*c / (x0+dx1)
        poly1 = c0 + c1*V1 + c2*V1*V1
        poly2 = d0 + d1*V2 + d2*V2*V2
        G1 = (jnp.exp(-0.5 * (V1/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V1/sigma/jnp.sqrt(2)))
        G2 = (jnp.exp(-0.5 * (V2/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V2/sigma/jnp.sqrt(2)))
        D1 = A   * G1/jnp.max(G1) + poly1
        D2 = A/R * G2/jnp.max(G2) + poly2
        D = jnp.concatenate([D1,D2])
        return D.mean(axis=1)
    
    @jit#The model for 2 lines in absorption.
    def line_model_2_abs(x_super, A=0, x0=0.0,sigma=1.0,alpha=0,c0=1.0,c1=1.0,c2=0.0,R=0, dx1=0.0,d0=0.0,d1=0.0,d2=0.0):
        """x goes in units of wavelength. Sigma goes in km/s.
        Alpha and sigma are the same for both components.
        This is an absorption line model, so it is multiplicative."""
        c = 299792.458 #km/s
        X1 = (x_super[0]-x0)
        X2 = (x_super[1]-(x0+dx1))
        V1 = X1*c / x0
        V2 = X2*c / (x0+dx1)
        poly1 = c0 + c1*V1 + c2*V1*V1
        poly2 = d0 + d1*V2 + d2*V2*V2
        G1 = (jnp.exp(-0.5 * (V1/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V1/sigma/jnp.sqrt(2)))
        G2 = (jnp.exp(-0.5 * (V2/sigma)**2)) * (1+jax.scipy.special.erf(alpha*V2/sigma/jnp.sqrt(2)))
        D1 = jnp.exp(-A   * G1/jnp.max(G1)) * poly1
        D2 = jnp.exp(-A/R * G2/jnp.max(G2)) * poly2
        D = jnp.concatenate([D1,D2])
        return D.mean(axis=1)
 
    @jit#The model for 1 line with 3 components.
    def line_model_1_triple(x_super, A1=0, x01=0.0,sigma1=1.0,alpha1=0,c0=0.0,c1=0.0,c2=0.0,A2=0.0,x02=0.0,sigma2=0.0,alpha2=0.0,A3=0.0,x03=0.0,sigma3=0.0,alpha3=0.0):
        """x goes in units of wavelength. Sigma goes in km/s. Allows for up to 3 components to be set."""
        c = 299792.458 #km/s
        X1 = (x_super-x01)
        X2 = (x_super-x02)
        X3 = (x_super-x03)
        V1 = X1*c/x01
        V2 = X2*c/x02
        V3 = X3*c/x03
        G1 = (jnp.exp(-0.5 * (V1/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V1/sigma1/jnp.sqrt(2)))
        G2 = (jnp.exp(-0.5 * (V2/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V2/sigma2/jnp.sqrt(2)))
        G3 = (jnp.exp(-0.5 * (V3/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V3/sigma3/jnp.sqrt(2)))
        D = A1 * G1/jnp.max(G1) + A2 * G2/jnp.max(G2) + A3 * G3/jnp.max(G3) + c0 + c1*V1 +c2*V1*V1
        return D.mean(axis=1)

    @jit#The model for 1 line with 3 components in absorption.
    def line_model_1_abs_triple(x_super, A1=0.0, x01=0.0,sigma1=1.0,alpha1=0,c0=1.0,c1=0.0,c2=0.0,A2=0.0,x02=0.0,sigma2=0.0,alpha2=0.0,A3=0.0,x03=0.0,sigma3=0.0,alpha3=0.0):
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
        G1 = (jnp.exp(-0.5 * (V1/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V1/sigma1/jnp.sqrt(2)))
        G2 = (jnp.exp(-0.5 * (V2/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V2/sigma2/jnp.sqrt(2)))
        G3 = (jnp.exp(-0.5 * (V3/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V3/sigma3/jnp.sqrt(2)))
        D = jnp.exp(-A1 * G1/jnp.max(G1)) * jnp.exp(-A2 * G2/jnp.max(G2)) * jnp.exp(-A3 * G3/jnp.max(G3)) * poly
        return D.mean(axis=1)
 
    @jit#The model for 2 lines. Up to 3 components.
    def line_model_2_triple(x_super, A1=0, x01=0.0,sigma1=1.0,alpha1=0,c0=1.0,c1=0.0,c2=0.0,R1=0,dx1=0.0,d0=0.0,d1=0.0,d2=0.0,A2=0.0,x02=0.0,sigma2=0.0,alpha2=0.0,R2=0.0,A3=0.0,x03=0.0,sigma3=0.0,alpha3=0.0,R3=0.0):
        """x goes in units of wavelength. Sigma goes in km/s.
        Alpha and sigma are the same for both components.
        This is an absorption line model, so it is multiplicative."""
        c = 299792.458 #km/s
        X11 = (x_super[0]-x01)
        X12 = (x_super[0]-x02)
        X13 = (x_super[0]-x03)

        X21 = (x_super[1]-(x01+dx1))
        X22 = (x_super[1]-(x02+dx1))
        X23 = (x_super[1]-(x03+dx1))

        V11 = X11*c / x01
        V12 = X12*c / x02
        V13 = X13*c / x03

        V21 = X21*c / (x01+dx1)
        V22 = X22*c / (x02+dx1)
        V23 = X23*c / (x03+dx1)

        poly1 = c0 + c1*V11 + c2*V11*V11
        poly2 = d0 + d1*V21 + d2*V21*V21

        G11 = (jnp.exp(-0.5 * (V11/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V11/sigma1/jnp.sqrt(2)))
        G12 = (jnp.exp(-0.5 * (V12/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V12/sigma2/jnp.sqrt(2)))
        G13 = (jnp.exp(-0.5 * (V13/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V13/sigma3/jnp.sqrt(2)))

        G21 = (jnp.exp(-0.5 * (V21/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V21/sigma1/jnp.sqrt(2)))
        G22 = (jnp.exp(-0.5 * (V22/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V22/sigma2/jnp.sqrt(2)))
        G23 = (jnp.exp(-0.5 * (V23/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V23/sigma3/jnp.sqrt(2)))

        D1 = A1 * G11/jnp.max(G11) + A2 * G12/jnp.max(G12) + A3 * G13/jnp.max(G13) + poly1
        D2 = A1/R1 * G21/jnp.max(G21) + A2/R2 * G22/jnp.max(G22) + A3/R3 * G23/jnp.max(G23) + poly2
        D = jnp.concatenate([D1,D2])
        return D.mean(axis=1)
    
    @jit#The model for 2 lines in absorption. Up to 3 components.
    def line_model_2_abs_triple(x_super, A1=0, x01=0.0,sigma1=1.0,alpha1=0,c0=1.0,c1=0.0,c2=0.0,R1=0,dx1=0.0,d0=0.0,d1=0.0,d2=0.0,A2=0.0,x02=0.0,sigma2=0.0,alpha2=0.0,R2=0.0,A3=0.0,x03=0.0,sigma3=0.0,alpha3=0.0,R3=0.0):
        """x goes in units of wavelength. Sigma goes in km/s.
        Alpha and sigma are the same for both components.
        This is an absorption line model, so it is multiplicative."""
        c = 299792.458 #km/s
        X11 = (x_super[0]-x01)
        X12 = (x_super[0]-x02)
        X13 = (x_super[0]-x03)

        X21 = (x_super[1]-(x01+dx1))
        X22 = (x_super[1]-(x02+dx1))
        X23 = (x_super[1]-(x03+dx1))

        V11 = X11*c / x01
        V12 = X12*c / x02
        V13 = X13*c / x03

        V21 = X21*c / (x01+dx1)
        V22 = X22*c / (x02+dx1)
        V23 = X23*c / (x03+dx1)

        poly1 = c0 + c1*V11 + c2*V11*V11
        poly2 = d0 + d1*V21 + d2*V21*V21

        G11 = (jnp.exp(-0.5 * (V11/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V11/sigma1/jnp.sqrt(2)))
        G12 = (jnp.exp(-0.5 * (V12/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V12/sigma2/jnp.sqrt(2)))
        G13 = (jnp.exp(-0.5 * (V13/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V13/sigma3/jnp.sqrt(2)))

        G21 = (jnp.exp(-0.5 * (V21/sigma1)**2)) * (1+jax.scipy.special.erf(alpha1*V21/sigma1/jnp.sqrt(2)))
        G22 = (jnp.exp(-0.5 * (V22/sigma2)**2)) * (1+jax.scipy.special.erf(alpha2*V22/sigma2/jnp.sqrt(2)))
        G23 = (jnp.exp(-0.5 * (V23/sigma3)**2)) * (1+jax.scipy.special.erf(alpha3*V23/sigma3/jnp.sqrt(2)))

        D1 = jnp.exp(-A1 * G11/jnp.max(G11)) * jnp.exp(-A2 * G12/jnp.max(G12)) * jnp.exp(-A3 * G13/jnp.max(G13)) * poly1
        D2 = jnp.exp(-A1/R1 * G21/jnp.max(G21)) * jnp.exp(-A2/R2 * G22/jnp.max(G22)) * jnp.exp(-A3/R3 * G23/jnp.max(G23)) * poly2
        D = jnp.concatenate([D1,D2])
        return D.mean(axis=1)
    #You can add more model functions with their associated parameters by following these as examples.
    

    #Switching between absorption or emission.
    if abs == False:
        if nc == 1:
            model_1 = line_model_1
            model_2 = line_model_2
        else:
            model_1 = line_model_1_triple
            model_2 = line_model_2_triple
    else:
        if nc == 1:
            model_1 = line_model_1_abs
            model_2 = line_model_2_abs
        else:
            model_1 = line_model_1_abs_triple
            model_2 = line_model_2_abs_triple



    if nterms == 1:
        def numpyro_model(predict=False):
            beta = get_beta()
            A = get_A()
            x0 = get_x0()
            sigma = get_sigma()   
            alpha=get_alpha()
            c0 = get_c0()
            c1 = get_c1()
            c2 = get_c2()


            model_spectrum = model_1(x_supers,A,x0,sigma,alpha,c0,c1,c2)

            if predict:
                numpyro.deterministic("model_spectrum", model_spectrum)
            numpyro.sample("obs", dist.Normal(loc=model_spectrum,scale=beta*YERR), obs=Y)

    if nterms == 2:
        def numpyro_model(predict=False):
            beta = get_beta()
            A = get_A()
            x0 = get_x0()
            sigma = get_sigma()   
            alpha=get_alpha()
            c0 = get_c0()
            c1 = get_c1()
            c2 = get_c2()
            R = get_R()
            dx = get_dx()
            d0 = get_d0()
            d1 = get_d1()
            d2 = get_d2()
            model_spectrum = model_2(x_supers,A,x0,sigma,alpha,c0,c1,c2,R,dx,d0,d1,d2)        
            if predict:
                numpyro.deterministic("model_spectrum", model_spectrum)
            numpyro.sample("obs", dist.Normal(loc=model_spectrum,scale=beta*YERR), obs=Y)


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
def test():
    import numpy.random
    import matplotlib.pyplot as plt
    import numpy as np
    noise = 0.02
    x1 = np.arange(580,581,0.01)
    x2 = np.arange(583,584,0.01)
    xs1 = supersample(x1)    
    xs2 = supersample(x2)    
    y1 = gaussian_skewed(xs1,-0.50,580.5,15.0,0.0,1.0,0.0,0.0)+numpy.random.normal(scale=noise,size=len(x1))
    y2 = gaussian_skewed(xs2,-0.25,583.5,15.0,0.0,1.0,0.0,0.0)+numpy.random.normal(scale=noise,size=len(x1))
    yerr1 = np.zeros_like(x1)+noise
    yerr2 = np.zeros_like(x2)+noise
    plt.errorbar(x1,y1,yerr=yerr1)
    plt.errorbar(x2,y2,yerr=yerr2)
    plt.title('Raw Gaussian profile')
    plt.show()


    #       beta,  A,    x0,   sigma,alpha,c0,   c1,   c2,  R,  dx1, d0,    d1,   d2
    lower = [0.5, -0.9, 580.3, 8.00, 0.0, 0.95, -2e-4, 0.0, 0.9, 2.8, 0.95, -2e-4, 0.0]
    upper = [2.0, -0.0, 580.7, 30.0, 0.0, 1.05,  2e-4, 0.0, 3.0, 3.2, 1.05,  2e-4, 0.0]
    # fit_lines([x1,x2],[y1,y2],[yerr1,yerr2],lower,upper,cpu_cores=3,oversample=5,progress_bar=True,nwarmup=1200,nsamples=500)


    y1 = gaussian_skewed_abs(xs1,3.0,580.5,15.0,0.0,1.0,0.0,0.0)+numpy.random.normal(scale=noise,size=len(x1))
    y2 = gaussian_skewed_abs(xs2,1.0,583.5,15.0,0.0,1.0,0.0,0.0)+numpy.random.normal(scale=noise,size=len(x1))
    yerr1 = np.zeros_like(x1)+noise
    yerr2 = np.zeros_like(x2)+noise
    plt.errorbar(x1,y1,yerr=yerr1)
    plt.errorbar(x2,y2,yerr=yerr2)
    plt.title('Absorption profile with A=tau_zero')
    plt.show()

    #       beta,  A,    x0,   sigma,alpha,c0,   c1,   c2,  R,  dx1, d0,    d1,   d2
    lower = [0.5, 0.00, 580.3, 8.00, 0.0, 0.95, -2e-4, 0.0, 1.0, 2.8, 0.95, -2e-4, 0.0]
    upper = [2.0, 10.0, 580.7, 30.0, 0.0, 1.05,  2e-4, 0.0, 6.0, 3.2, 1.05,  2e-4, 0.0]
    fit_lines([x1,x2],[y1,y2],[yerr1,yerr2],lower,upper,cpu_cores=3,oversample=5,progress_bar=True,nwarmup=1200,nsamples=500,abs=True)

