import csv
import pandas as pd
import numpy as np

from pypws.enums import ResultCode

from py_lopa.calcs import helpers
from py_lopa.calcs.consts import Consts
from py_lopa.data.tables import Tables
from py_lopa.model_interface import Model_Interface
from py_lopa.classes.solver import Solver

cd = Consts().CONSEQUENCE_DATA
cheminfo = helpers(Tables().CHEM_INFO)
test_cases_df = pd.read_csv('heuristing_testing_output.csv')

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
    m_io.inputs['chem_mix'] = row['chem_mix']
    m_io.inputs['temp_deg_c'] = row['temp_c']
    m_io.inputs['pressure_psig'] = 100
    m_io.inputs['release_elevation_m'] = 0
    m_io.inputs['flash_fire'] = False
    m_io.inputs['max_hole_size_in'] = 5
    m_io.inputs['bldg_hts_low_med_high'] = [0,0,0]
    m_io.inputs['storage_mass_kg'] = 10000 / 2.2
    return m_io

def parse_m_io(m_io):
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

def model_run_for_solver(x0, args):
    target_input = args['target_input']
    m_io = args['m_io']
    target_category = args['target_category']
    m_io.inputs[target_input] = x0
    res = m_io.run()
    if res != ResultCode.SUCCESS:
        return None
    output = parse_m_io(m_io)
    ans = output[f'{target_category}_{cd.IMPACT_DISTANCE_M}']
    return ans


def solve_for_dist_at_conc(m_io):
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
                print(f"{m_io.inputs['chem_mix']} could not be solved for {target_category} over {target_input}.  Error: {e.args} ")
                continue
    return output

def main():
    output = []
    for _, row in test_cases_df.iterrows():
        starting_inputs['temp_deg_c'] = row['temp_c']
        m_io = get_m_io(row)
        chem_results = solve_for_dist_at_conc(m_io)
        output.extend(chem_results)
        if len(output) == 0:
            continue
        try:
            with open('sensitivity_output.csv', newline='') as csv_file:
                writer = csv.DictWriter(f=csv_file, fieldnames=list(chem_results[-1].keys()))
                writer.writerows(output)
                output = []
        except Exception as e:
            print(f"Could not write results to file.  will attempt to write next time.  current results in buffer: \n\n\n{output}")


