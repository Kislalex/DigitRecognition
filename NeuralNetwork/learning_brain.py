from .brain import Brain
import numpy as np
import time
from multiprocessing import Pool
from functools import partial
import os

Eps = 0.0001


def quadratic_gradient(output_, goal_):
    df = output_ - goal_
    return df


def quadratic_energy(output_, goal_):
    df = output_ - goal_
    return np.linalg.norm(df)


def get_gradient_from_test(brain, energy_gradient, learinig_io):
    brain_output = brain.compute(learinig_io[0])
    df = energy_gradient(brain_output[0], learinig_io[1])
    _, db = brain.compute_gradient(df, learinig_io[0])
    return db


def energy_from_test(brain, energy, learinig_io):
    brain_output = brain.compute(learinig_io[0])
    return energy(brain_output[0], learinig_io[1])


def compute_test_error(brain, learinig_batch, energy):
    energies = []
    find_energy = partial(energy_from_test, brain, energy)
    with Pool(16) as p:
        work = p.map_async(find_energy, learinig_batch)
        p.close()
        p.join()
        energies = work.get()
    return sum(energies)


def compute_batch_gradient(brain, learinig_batch, energy_gradient):
    dt = 1.0 / len(learinig_batch)
    find_gradient = partial(get_gradient_from_test, brain, energy_gradient)
    gradients = []
    with Pool(16) as p:
        work = p.map_async(find_gradient, learinig_batch)
        p.close()
        p.join()
        gradients = work.get()
    collective_gradient = gradients[0]
    for gradient in gradients[1:]:
        collective_gradient = collective_gradient + gradient
    return dt * collective_gradient


def find_the_best_brain_update(brain, learinig_batch, energy_gradient, energy, scales):
    min_error = 100000000000000.0
    best_brain = brain.copy()
    gradient = compute_batch_gradient(brain, learinig_batch, energy_gradient)
    for x in scales:
        new_brain = brain + (-x) * gradient
        new_error = compute_test_error(new_brain, learinig_batch, energy)
        if new_error < min_error:
            min_error = new_error
            best_brain = new_brain.copy()
    return best_brain


def learning_brain_by_g_d(
    brain,
    learinig_batch,
    step_number=100,
    energy_gradient=quadratic_gradient,
    energy=quadratic_energy,
    scales=(1.0,),
):
    previous_error = 0
    delta_error = 0
    begin = time.time()
    for step in range(step_number):
        brain = find_the_best_brain_update(
            brain, learinig_batch, energy_gradient, energy, scales
        )
        error = compute_test_error(brain, learinig_batch, energy)
        delta_error = previous_error - error
        previous_error = error
        print("time : {}".format(time.time() - begin))
        begin = time.time()
        print("step     : ", step)
        print("total err: ", error)
        print("avg   err: ", error / len(learinig_batch))
        print("del   err: ", delta_error)
    return brain
