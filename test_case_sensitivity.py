import csv
import pandas as pd
import numpy as np

from pypws.enums import ResultCode

from py_lopa.calcs import helpers
from py_lopa.calcs.consts import Consts
from py_lopa.data.tables import Tables
from py_lopa.model_interface import Model_Interface
from py_lopa.classes.solver import Solver
from py_lopa.calcs.flattening import Flattening

cd = Consts().CONSEQUENCE_DATA
cheminfo = helpers.get_dataframe_from_csv(Tables().CHEM_INFO)
test_cases_df = pd.read_csv('heuristing_testing_output.csv')

moderate_conc_ppm = 1000  # test cases run for chemicals with ERPG-3 at 1000 ppm

starting_inputs = {
    'storage_mass_kg' : 10000 / 2.2,
    'pressure_psig' : 100,
    'release_elevation_m' : 0,
    'max_hole_size_in' : 5,
}

targ_dists = {
    cd.CAT_SERIOUS : 100, # distance to control rooms
    cd.CAT_MODERATE : 200, # distance to offsite
}

def get_m_io(row):
    m_io = Model_Interface()
    m_io.set_inputs_as_arguments()
    m_io.inputs['chemical_mix'] = row['chem_mix']
    m_io.inputs['temp_deg_c'] = row['temp_c']
    m_io.inputs['pressure_psig'] = 100
    m_io.inputs['release_elevation_m'] = 0
    m_io.inputs['flash_fire'] = False
    m_io.inputs['max_hole_size_in'] = 5
    m_io.inputs['bldg_hts_low_med_high'] = [0,0,0]
    m_io.inputs['storage_mass_kg'] = 10000 / 2.2
    return m_io

def parse_m_io(m_io, target_category):
    phast_disp_dict = m_io.mc.phast_disp
    targ_conc_volf = moderate_conc_ppm * 1e-6
    if target_category == cd.CAT_SERIOUS:
        targ_conc_volf *= 10 # 10xERPG-3 for Serious
    for rel_dur_sec, pd_at_dur in phast_disp_dict.items():
        for wx, pd_at_dur_at_wx in pd_at_dur[rel_dur_sec].items():
            for haz, pd_at_dur_at_wx_at_haz in pd_at_dur_at_wx[wx].items():
                p_disp = pd_at_dur_at_wx_at_haz[haz]
                flattening = Flattening(conc_pfls=p_disp.conc_profiles)
                dist_m = flattening.calc_max_dist_at_conc(targ_conc_volf=targ_conc_volf, min_ht_m=0, max_ht_m=6)

    return dist_m

def model_run_for_solver(x0, args):
    target_input = args['target_input']
    m_io = args['m_io']
    target_category = args['target_category']
    m_io.inputs[target_input] = x0
    res = m_io.run()
    if res != ResultCode.SUCCESS:
        return None
    dist_m = parse_m_io(m_io, target_category=target_category)
    return dist_m

def solve_for_dists_at_concs(m_io):
    output = []
    args = {
        'm_io' : m_io,
    }
    for target_category in targ_dists.keys():
        target_distance = targ_dists[target_category]
        args['target_category'] = target_category
        for target_input in starting_inputs.keys():
            args['target_input'] = target_input
            x0 = starting_inputs[target_input]
            solver = Solver(arg_to_vary=x0, fxn_to_solve=model_run_for_solver, args=args, target=target_distance)
            try:
                if solver.solve():
                    output.append({
                        'chem_mix': m_io['chem_mix'],
                        'target_category': target_category,
                        'target_input' : target_input,
                        'value': solver.answer
                    })
            except Exception as e:
                print(f"{m_io.inputs['chemical_mix']} could not be solved for {target_category} over {target_input}.  Error: {e.args} ")
                continue
    return output

def main():
    output = []
    for _, row in test_cases_df.iterrows():
        starting_inputs['temp_deg_c'] = row['temp_c']
        m_io = get_m_io(row)
        chem_results = solve_for_dists_at_concs(m_io)
        output.extend(chem_results)
        if len(output) == 0:
            continue
        try:
            with open('sensitivity_output.csv', 'a', newline='') as csv_file:
                writer = csv.DictWriter(f=csv_file, fieldnames=list(chem_results[-1].keys()))
                writer.writerows(output)
                output = []
        except Exception as e:
            print(f"Could not write results to file.  Will attempt to write next time.  \ncurrent results in buffer: \n{output}")

if __name__ == '__main__':
    main()