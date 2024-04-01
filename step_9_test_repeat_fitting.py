# This is to tackle issue 78.
# The idea is that repeat calls to MCMC save a compilation step. Very interesting if true.


if __name__ == "__main__":
    import numpyro
    from numpyro.infer import MCMC, NUTS, Predictive
    from numpyro import distributions as dist
    # Set the number of cores on your machine for parallelism:
    cpu_cores = 1
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
    from fitting import supersample
    import numpy.random

    #The model for 1 line. ALSO CHANGE IN THE NUMPY CODE BELOW. THIS ONE IS THE NON-JITTED VERSION.
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
    


    A1 = 1.0
    mu1= 580.0
    sigma1=20.0
    A2 = 3.0
    mu2= 580.1
    sigma2=10.0

    N = 0.02
    X = np.arange(579.7,580.3,0.01)
    noise = numpy.random.normal(size=len(X),scale=N)
    yerr  = X*0.0+N
    X_super = supersample(X,f=20)
    Y1 = line_model(X_super,A=A1,mu=mu1,sigma=sigma1) + noise
    Y2 = line_model(X_super,A=A2,mu=mu2,sigma=sigma2) + noise





    def numpyro_model(x_in,y_in,predict=False):
            A = numpyro.sample('A',dist.Uniform(low=-10,high=10))
            mu = numpyro.sample('mu',dist.Uniform(low=579.7,high=580.3))
            sigma = numpyro.sample('sigma',dist.Uniform(low=5,high=50))


            model_spectrum = line_model(x_in, A=A, 
                                        mu=mu,
                                        sigma=sigma)
            if predict:
                numpyro.deterministic("model_spectrum", model_spectrum)
            numpyro.sample("obs", dist.Normal(loc=model_spectrum,scale=yerr), obs=y_in)



    rng_seed = 0
    rng_keys = split(PRNGKey(rng_seed), cpu_cores)
    if cpu_cores == 1:
         rng_keys = PRNGKey(rng_seed)
    sampler = NUTS(numpyro_model,dense_mass=True)

    mcmc = MCMC(sampler, 
                     num_warmup=1000, 
                     num_samples=200, 
                     num_chains=cpu_cores,progress_bar=True,jit_model_args=True)
    
    import tayph.util as ut
    t1 = ut.start()
    mcmc.run(rng_keys,X_super,Y1)
    ut.end(t1)

    t1 = ut.start()
    mcmc.run(rng_keys,X_super,Y2)
    ut.end(t1)


    result = arviz.from_numpyro(mcmc)
    corner(result,quiet=True,show_titles=True)

    plt.figure()
    plt.plot(X,Y1)
    plt.plot(X,Y2)
    plt.show()


    # The answer appears to be that it works, but only for single chains. My use-case then probably does not benefit.