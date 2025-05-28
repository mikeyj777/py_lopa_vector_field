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
test_cases_df = pd.read_csv('sensitivity_testing_inputs.csv')

moderate_conc_ppm = 1000  # test cases run for chemicals with ERPG-3 at 1000 ppm

def multiplier(var_in, factor, inverse = False):
    ans = var_in * factor
    if inverse:
        ans = var_in / factor
    return ans

def adder(var_in, factor, inverse = False):
    ans = var_in + factor
    if inverse:
        ans = var_in - factor
    return ans

starting_inputs_list = [
    {   
        'parameter': 'storage_mass_kg',
        'initial_value': 10000 / 2.2,
        'lower_bound': 10,
        'upper_bound': 10000,
        'increment_operation': multiplier,
        'increment_factor': 1.1,
    },
    {   
        'parameter' : 'pressure_psig',
        'initial_value' : 100,
        'lower_bound' : 0.1,
        'upper_bound' : 1000,
        'increment_operation': multiplier,
        'increment_factor': 1.1,
    },
    {   
        'parameter' : 'release_elevation_m',
        'initial_value': 3,
        'lower_bound' : 0,
        'upper_bound' : 6,
        'increment_operation': adder,
        'increment_factor': 1,
    },
    {   
        'parameter': 'max_hole_size_in',
        'initial_value' : 5,
        'lower_bound' : 0.1,
        'upper_bound' : 50,
        'increment_operation': multiplier,
        'increment_factor': 1.1,

    },
    {
        'parameter' : 'temp_deg_k',
        'initial_value' : None,
        'lower_bound' : -150+273.15,
        'upper_bound' : 800+273.15,
        'increment_operation': multiplier,
        'increment_factor': 1.1,
    }
]

targ_dists = {
    cd.CAT_SERIOUS : 100, # distance to control rooms
    cd.CAT_MODERATE : 200, # distance to offsite
}

def get_m_io(row):
    m_io = Model_Interface()
    m_io.set_inputs_as_arguments()
    m_io.inputs['chemical_mix'] = helpers.get_data_from_pandas_series_element(row['chem_mix'])
    m_io.inputs['flash_fire'] = False
    m_io.inputs['bldg_hts_low_med_high'] = [0,0,0]
    for starting_input_dict in starting_inputs_list:
        adding_factor = 0
        param = starting_input_dict['parameter']
        if param == 'temp_deg_k':
            param = 'temp_deg_c'
            adding_factor = -273.15
            starting_input_dict['initial_value'] = helpers.get_data_from_pandas_series_element(row['temp_c']) + 273.15
        value = starting_input_dict['initial_value'] + adding_factor
        
        m_io.inputs[param] = value
    
    return m_io

def parse_m_io(m_io):
    phast_disp_dict = m_io.mc.phast_disp
    targ_conc_volf = moderate_conc_ppm * 1e-6
    for target_category in targ_dists.keys():
        for rel_dur_sec, pd_at_dur in phast_disp_dict.items():
            for wx, pd_at_dur_at_wx in pd_at_dur.items():
                for haz, pd_at_dur_at_wx_at_haz in pd_at_dur_at_wx.items():
                    p_disp = pd_at_dur_at_wx_at_haz
                    flattening = Flattening(conc_pfls=p_disp.conc_profiles)
                    moderate_dist_m = flattening.calc_max_dist_at_conc(targ_conc_volf=targ_conc_volf, min_ht_m=0, max_ht_m=6)
                    serious_dist_m = flattening.calc_max_dist_at_conc(targ_conc_volf=targ_conc_volf*10, min_ht_m=0, max_ht_m=6)

    return {
        cd.CAT_MODERATE : moderate_dist_m,
        cd.CAT_SERIOUS : serious_dist_m,
    }

def model_run_for_solver(x0, args):
    parameter_to_vary = args['parameter_to_vary']
    row = args['row']
    m_io = get_m_io(row)
    adding_factor = 0
    if parameter_to_vary == 'temp_deg_k':
        parameter_to_vary = 'temp_deg_c'
        adding_factor = -273.15
    m_io.inputs[parameter_to_vary] = x0 + adding_factor
    res = m_io.run()
    if res != ResultCode.SUCCESS:
        return None
    dists_m = parse_m_io(m_io)
    return dists_m

def get_sensitivity_around_init_values(row):
    output = []
    args = {
        'row' : row,
    }
    
    for parameter_data in starting_inputs_list:
        args['parameter_data'] = parameter_data
        parameter_to_vary = parameter_data['parameter']
        args['parameter_to_vary'] = parameter_to_vary
        x0 = parameter_data['initial_value']
        xlow = x0 - 1
        xhigh = x0 + 1
        for x in [xlow, x0, xhigh]:
            try:
                dists_m = model_run_for_solver(x0=x, args=args)
                output.append({
                    'chem_mix': row['chem_mix'],
                    'parameter_to_vary' : parameter_to_vary,
                    'parameter_value': x,
                    'moderate_dist_m': dists_m[cd.CAT_MODERATE],
                    'serious_dist_m': dists_m[cd.CAT_SERIOUS],
                })
                print(output)
            except Exception as e:
                    print(f"{row['chem_mix']} could not be solved over {parameter_to_vary} at {x}.  Error: {e.args} ")
                    continue
            
            try:
                with open('sensitivity_output.csv', 'a', newline='') as csv_file:
                    writer = csv.DictWriter(f=csv_file, fieldnames=list(output[-1].keys()))
                    writer.writerows(output)
                    output = []
            except Exception as e:
                print(f"Could not write results to file.  Will attempt to write next time.  \ncurrent results in buffer: \n{output}")
    return output


def main():
    # output = []
    for _, row in test_cases_df.iterrows():
        get_sensitivity_around_init_values(row)
        
if __name__ == '__main__':
    main()