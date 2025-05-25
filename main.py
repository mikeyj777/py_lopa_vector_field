import csv
import copy
import pickle
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import pandas as pd

from pypws.enums import ResultCode

from py_lopa.calcs import helpers
from py_lopa.calcs.consts import Consts
from py_lopa.data.tables import Tables
from py_lopa.phast_io.phast_prep import prep_state, flash_calc
from py_lopa.calcs.thermo_pio import vpress_pa_and_vapor_phase_comp_and_component_vapor_pressures
from py_lopa.model_interface import Model_Interface

cd = Consts().CONSEQUENCE_DATA

starting_idx = -1
try:
    with open('curr_idx.txt', 'r') as idx_file:
        starting_idx = int(idx_file.read())
except:
    pass

cheminfo = helpers.get_dataframe_from_csv(Tables().CHEM_INFO)

# load materials pickle
mat_and_props_list = helpers.load_pickled_object('data/materials.pickle')

elevs_m = np.linspace(0,6,4)
orientations = [0, 90]
pressures_psig = np.linspace(0,500, 101)
pressures_psig[0] = 1
release_kmoles = [100, 300, 1000, 3000, 10000, 30000]
hole_sizes_in = np.linspace(0.1,10,11)

data_to_write = []
# iterate across df to generate list of cas_no, temperature, moles, hole size, elevation, orientation
num_runs = len(mat_and_props_list) * len(pressures_psig) * 21 * len(release_kmoles) * len(elevs_m) * len(orientations)
idx = -1
t0 = dt.now()
t1 = dt.now()
time_per_run_min = 1e-5
header_written = False

def get_m_io(row) -> Model_Interface:
    m_io = Model_Interface()
    m_io.material = row['material']
    m_io.flashresult = row['flashresult']
    m_io.inputs['chemical_mix'] = [row['cas_no']]
    m_io.inputs['composition'] = [1]
    m_io.inputs['pressure_psig'] = row['press_psig']
    m_io.inputs['temp_deg_c'] = row['temp_k'] - 273.15
    m_io.inputs['storage_mass_kg'] = row['release_mass_kg']
    m_io.inputs['release_elevation_m'] = row['elev_m']
    m_io.inputs['release_angle_degrees_from_horizontal'] = row['orientation']
    m_io.inputs['flash_fire'] = False
    m_io.inputs['inhalation'] = True
    m_io.inputs['max_hole_size_in'] = row['hole_size_in']
    m_io.inputs['bldg_hts_low_med_high'] = [0 , 0, 0]
    m_io.inputs['use_multicomponent_method'] = False

    return m_io

def process_m_io_return_areas_and_distances_list_of_dicts(m_io:Model_Interface):
    res = m_io.run()
    if res != ResultCode.SUCCESS:
        return None
    ca_dict = m_io.mc.ca_dict

    # self.ca_dict[release_duration_sec][wx][haz] = copy.deepcopy(ca.consequence_list)
    output = {}
    for rel_dur_sec, ca_data_at_dur in ca_dict.items():
        for wx, ca_at_dur_at_wx in ca_data_at_dur[rel_dur_sec].items():
            for haz, ca_at_dur_at_wx_at_haz in ca_at_dur_at_wx[wx].items():
                ca = ca_at_dur_at_wx_at_haz[haz]
                conseq_list = ca.conseq_list
                for con_dict in conseq_list:
                    if con_dict[cd.CAT_TITLE] in [cd.CAT_SERIOUS, cd.CAT_MODERATE]:
                        output[f'{con_dict[cd.CAT_TITLE]}_{cd.IMPACT_DISTANCE_M}'] = con_dict[cd.IMPACT_DISTANCE_M]
                        output[f'{con_dict[cd.CAT_TITLE]}_{cd.IMPACT_AREA_M2}'] = con_dict[cd.IMPACT_AREA_M2]
                        
    
    return output


def disp_model():
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
        chem_name = row['chem_name']
        
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
                row_inner_inner['flash_calc'] = pd.NA
                if starting_idx < idx:
                    try:
                        flashresult = flash_calc(state, material=material)
                        row_inner_inner['flashresult'] = flashresult
                    except Exception as e:
                        print(f'{chem_name} could not creat flash at {press_psig} psig and {temp_k} deg k')
                        continue
                row_inner_inner['temp_k'] = temp_k
                row_inner_inner['vp_pa'] = vp
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
                            if starting_idx >= idx:
                                continue
                            row_inner_inner_inner_inner_inner['orientation'] = orientation
                            for hole_size_in in hole_sizes_in:
                                row_inner_inner_inner_inner_inner_inner = copy.deepcopy(row_inner_inner_inner_inner_inner)
                                row_inner_inner_inner_inner_inner_inner['hole_size_in'] = hole_size_in
                                try:
                                    with open('curr_idx.txt', 'w') as idx_file_out:
                                        idx_file_out.write(str(idx))
                                    m_io = get_m_io(row_inner_inner_inner_inner_inner_inner)
                                    results = process_m_io_return_areas_and_distances_list_of_dicts(m_io)
                                    results['idx'] = idx
                                    row_inner_inner_inner_inner_inner_inner.update(results)
                                    data_to_write.append(row_inner_inner_inner_inner_inner_inner)
                                    with open('model_run_output.csv', 'a', newline='') as f:
                                        writer = csv.DictWriter(f, fieldnames=list(row_inner_inner_inner_inner_inner_inner.keys()))
                                        if not header_written:
                                            if starting_idx < 0:
                                                writer.writeheader()
                                            header_written = True
                                        writer.writerows(data_to_write)
                                        data_to_write = []

                                except Exception as e:
                                    print(f'could not store model idx {idx}.  Error: {e.args}')
                                t1 = dt.now()
                                tot_time_time_delta = t1 - t0
                                tot_time_sec = tot_time_time_delta.total_seconds()
                                if tot_time_sec != 0:
                                    time_per_run_min = idx / tot_time_sec / 60
                            remaining_runs = num_runs - idx - 1
                            remaining_minutes = time_per_run_min * remaining_runs
                            etc = dt.now() + timedelta(minutes=remaining_minutes)
                            print(f'{idx} of {num_runs}.  ')
                            
apple = 1