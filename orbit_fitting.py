import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import astropy.constants as const
jax.config.update("jax_enable_x64", True)
solmass = float(const.M_sun.value)
G = const.G.value





@jit
def kepler_starter(mean_anom, ecc):
    ome = 1 - ecc
    M2 = jnp.square(mean_anom)
    alpha = 3 * jnp.pi / (jnp.pi - 6 / jnp.pi)
    alpha += 1.6 / (jnp.pi - 6 / jnp.pi) * (jnp.pi - mean_anom) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * mean_anom
    q = 2 * alphad * ome - M2
    q2 = jnp.square(q)
    w = jnp.square(jnp.cbrt(jnp.abs(r) + jnp.sqrt(q2 * q + r * r)))
    return (2 * r * w / (jnp.square(w) + w * q + q2) + mean_anom) / d

@jit
def kepler_refiner(mean_anom, ecc, ecc_anom):
    ome = 1 - ecc
    sE = ecc_anom - jnp.sin(ecc_anom)
    cE = 1 - jnp.cos(ecc_anom)
    f_0 = ecc * sE + ecc_anom * ome - mean_anom
    f_1 = ecc * cE + ome
    f_2 = ecc * (ecc_anom - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24)
    return ecc_anom + dE

@jit
@jnp.vectorize
def kepler_solver_impl(mean_anom, ecc):
    mean_anom = mean_anom % (2 * jnp.pi)
    # We restrict to the range [0, pi)
    high = mean_anom > jnp.pi
    mean_anom = jnp.where(high, 2 * jnp.pi - mean_anom, mean_anom)
    # Solve
    ecc_anom = kepler_starter(mean_anom, ecc)
    ecc_anom = kepler_refiner(mean_anom, ecc, ecc_anom)
    # Re-wrap back into the full range
    ecc_anom = jnp.where(high, 2 * jnp.pi - ecc_anom, ecc_anom)
    return ecc_anom   




@jit
def kepler_III(a,Ms):
    """Return period, everything in SI units."""
    return(jnp.sqrt(4*np.pi**2 *a**3 / (G*Ms)))

@jit
def keplerian(t_days,a_au=1.0,e=0.0,M_Msun=1.0):
    """
    This calculates the x,y positions and velocities of a keplerian orbit given orbital elements.
    It uses the example from https://jax.exoplanet.codes/en/latest/tutorials/core-from-scratch/
    And the theory as described here: https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Position_as_a_function_of_time (d Jul 1 2024)
     
    Input:
    t_days a time axis in days since periastron passage.
    a_au the semi-major axis in au.
    M_Msun the stellar mass in solar masses.
     
    
    Output:
    r,T,x,y,vx,vy
    Radial distance and angle (true anomaly) to describe the position of the object in polar coordinates,
    the position x and y in carthesian coordinates,
    the velocity vector vx, vy in carthesian coordinates.

    Note that this formulation has not yet rotated the orbit to a particular reference frame.
    """

    #System definition
    Ms = M_Msun*solmass #Stellar mass in kg
    a = a_au * const.au.value #AU in m
 
    t = (t_days)*24.0*3600.0 


    #Now assuming a closed elliptical orbit:
    P = kepler_III(a,Ms)

    n = 2*np.pi / P
    M = n*t #Mean anomaly
    E = kepler_solver_impl(M,e) #Eccentric anomaly
    b = jnp.sqrt((1+e)/(1-e))
    T = 2 * jnp.arctan(b * jnp.tan(E/2) ) #True anomaly
    r = a * (1 - e * jnp.cos(E))

    x,y = r*jnp.cos(T),r*jnp.sin(T)
    vx,vy = jnp.sqrt(G*Ms*a)/r * -jnp.sin(E) , jnp.sqrt(G*Ms*a)/r * jnp.sqrt(1-e**2) * jnp.cos(E)

    return(r,T,x,y,vx,vy)




@jit
def rot_orbital_elements(x,y,z,inc,omega,Omega):
    """This rotates an orbit, or its velocity vector, to a reference frame with orbital element angles
    inclination, argument of periastron and longitude of ascending node. All angles in radians. 

    About the coordinate system:
    For both omegas equal to zero, inclination acts along the x-axis, so that's to be understood to be inclination as being an observer.
    So the x-z plane is the plane of the sky. This makes negative y in the direction of the observer. So vy is the radial velocity.
    Making positive inclinations tilt the orbit so that it passes in front of the star in positive z direction, requires that big omega is set to 180 degrees.
    Making an eccentric orbit with the periastron pointing towards you requires that small omega is set to 90 degrees. 
    
    For i = 0, big and small omega are degenerate."""
    # r,T,x,y,vx,vy = keplerian(t_days,a_au,e,M_Msun)
    # z = r*0.0
    # vz = r*0.0
    #Bunch of rotation matrices:
    R_w = jnp.array([[jnp.cos(omega),jnp.sin(omega),0],[-jnp.sin(omega),jnp.cos(omega),0],[0,0,1]])
    R_i = jnp.array([[1,0,0],[0,jnp.cos(inc),jnp.sin(inc)],[0,-jnp.sin(inc),jnp.cos(inc)]])
    R_W = jnp.array([[jnp.cos(Omega),jnp.sin(Omega),0],[-jnp.sin(Omega),jnp.cos(Omega),0],[0,0,1]])

    return((jnp.array([x,y,z]).T @ R_w @ R_i @ R_W).T)
