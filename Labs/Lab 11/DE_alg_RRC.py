#https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/

import numpy as np

# Create global variables (1D arrays) for the measured data
x = np.empty(1)
y = np.empty(1)
Vs = 1

# DE algorithm
def de(fobj, bounds, mut, crossp, popsize, its):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


# Evaluates the RRC model for parameters given in the w vector.
def fmodel(x, w):
    global Vs
    x  = np.asarray(x)
    R1 = w[0]
    R2 = w[1]
    C  = w[2]

    alpha = R1/R2+1
    return Vs/alpha*(1-np.exp(-alpha/R1/C*x))


# A cost function for the optimization.
# Returns root mean square of the error (a difference between the model and experimental data
def rmse(w): 
    y_pred = fmodel(measured_x, w)
    return np.sqrt(sum((measured_y - y_pred)**2) / len(y))
    

def main(x, y, cluster):
    global measured_x
    global measured_y
    global Vs

    # Format the XY data into numpy arrays
    measured_x = np.asarray(x)
    measured_y = np.asarray(y)

    # Extract the DE parameters from the cluster
    popsize = cluster[0]
    its =     cluster[1]
    mut =     cluster[2]
    crossp =  cluster[3]
    bounds =  cluster[4]
    Vs     =  cluster[5]

    # Do differential evolution. It returns a generator datatype.
    result = list(de(rmse, bounds, mut, crossp, popsize, its))
    result = list(zip(*result))
    params = np.stack(list(result[0]), axis=0)
    err = list(result[1])
    opt_curve = np.asarray([fmodel(measured_x,ind) for ind in result[0]])
    print(opt_curve.shape)

    # Return parameter vectors for each iteration (a 2D array)
    return (params, err, opt_curve)

#x = np.linspace(0, 5, 20)
#y = fmodel(x,[100,200,0.01]) + np.random.normal(0, 0.5, 20)
#ret = main(x, y, (10,150,0.3,0.5,[(1,5),(1,200),(100,1E6),(1E-9,0)]))
#print(ret)