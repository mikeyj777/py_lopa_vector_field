import copy
import pickle
from datetime import datetime as dt
import numpy as np
import pandas as pd

from py_lopa.calcs import helpers
from py_lopa.data.tables import Tables
from py_lopa.phast_io.phast_prep import prep_state, flash_calc
from py_lopa.calcs.thermo_pio import vpress_pa_and_vapor_phase_comp_and_component_vapor_pressures

cheminfo = helpers.get_dataframe_from_csv(Tables().CHEM_INFO)

# load materials pickle
mat_and_props_list = helpers.load_pickled_object('data/materials.pickle')

elevs_m = np.linspace(0,6,4)
orientations = [0, 90]
pressures_psig = np.linspace(0,500, 101)
pressures_psig[0] = 1
release_kmoles = [100, 300, 1000, 3000, 10000, 30000]

output = []
# iterate across df to generate list of cas_no, temperature, moles, hole size, elevation, orientation
num_runs = len(pressures_psig) * 21 * len(release_kmoles) * len(elevs_m) * len(orientations)
idx = 0
t0 = dt.now()
for row in mat_and_props_list:

    row_out = copy.deepcopy(row)
    # iterate across temp (-100 to +100 deg above nbp)
    material = row['material']
    nbp_deg_k = max(200, row['nbp_deg_k'])
    cas_no = row['cas_no']
    mw = helpers.get_mw(cas_no, cheminfo=cheminfo)
    row_out['mw'] = mw
    min_t = max(100, nbp_deg_k-100)
    max_t = max(400, nbp_deg_k+100)
    temps_k = np.linspace(min_t, max_t, 21)
    
    #  mixture_cas_nos = []
    # if 'mixture_cas_nos' in args:
    #     mixture_cas_nos = args['mixture_cas_nos']
    
    # mixture_molfs = []
    # if 'mixture_molfs' in args:
    #     mixture_molfs = args['mixture_molfs']
    
    # cheminfo = None
    # if 'cheminfo' in args:

    args_for_vp_calc = {
        'cheminfo': cheminfo,
        'mixture_cas_nos': [cas_no],
        'mixture_molfs': [1],
    }

    # calculate vp
    vps = [vpress_pa_and_vapor_phase_comp_and_component_vapor_pressures(temp_k=temp_k, args=args_for_vp_calc)['vpress_pa'] for temp_k in temps_k]
    vps = np.array([max(0, vp) for vp in vps])

    # prep state
    for press_psig in pressures_psig:
        row_inner = copy.deepcopy(row_out)
        row_inner['press_psig'] = press_psig
        press_pa = (press_psig + 14.6959) * 101325 / 14.6959
        for temp_k, vp in zip(temps_k, vps):
            row_inner_inner = copy.deepcopy(row_inner)
            state = prep_state(press_pa=press_pa, temp_K=temp_k, use_multicomponent_modeling=False)
            flash = flash_calc(state, material=material, return_flash_calc=True)
            row_inner_inner['press_psig'] = press_psig
            row_inner_inner['temp_k'] = temp_k
            row_inner_inner['vp_pa'] = vp
            row_inner_inner['flash_calc'] = flash
            for release_kmol in release_kmoles:
                row_inner_inner_inner = copy.deepcopy(row_inner_inner)
                release_mass_kg = release_kmol * mw
                row_inner_inner_inner['release_mass_kg'] = release_mass_kg
                for elev_m in elevs_m:
                    row_inner_inner_inner_inner = copy.deepcopy(row_inner_inner_inner)
                    row_inner_inner_inner_inner['elev_m'] = elev_m
                    row_inner_inner_inner_inner_inner = copy.deepcopy(row_inner_inner_inner_inner)
                    for orientation in orientations:
                        idx += 1
                        row_inner_inner_inner_inner_inner['orientation'] = orientation
                        output.append(row_inner_inner_inner_inner_inner)
                        t1 = dt.now()
                        tot_time_time_delta = t1 - t0
                        tot_time_sec = tot_time_time_delta.total_seconds()
                        time_per_run_min = idx / tot_time_sec / 60
            with open('data_for_models.pickle', 'wb') as f:
                pickle.dump(output, f, pickle.DEFAULT_PROTOCOL)
            print(f'idx {idx} / {num_runs}.  est time remaining:  {(num_runs-idx) * time_per_run_min} minutes.  ')

apple = 1