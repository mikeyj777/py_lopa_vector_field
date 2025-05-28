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
test_cases_df = pd.read_csv('heuristing_testing_output_acetaldehyde.csv')

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
    # {   
    #     'parameter': 'storage_mass_kg',
    #     'initial_value': 10000 / 2.2,
    #     'lower_bound': 10,
    #     'upper_bound': 10000,
    #     'increment_operation': multiplier,
    #     'increment_factor': 1.1,
    # },
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
    # cd.CAT_SERIOUS : 100, # distance to control rooms
    cd.CAT_MODERATE : 200, # distance to offsite
}

def get_m_io(row):
    m_io = Model_Interface()
    m_io.set_inputs_as_arguments()
    m_io.inputs['chemical_mix'] = helpers.get_data_from_pandas_series_element(row['chem_mix'])
    m_io.inputs['flash_fire'] = False
    m_io.inputs['bldg_hts_low_med_high'] = [0,0,0]
    m_io.inputs['storage_mass_kg'] = 10000 / 2.2
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

def parse_m_io(m_io, target_category):
    phast_disp_dict = m_io.mc.phast_disp
    targ_conc_volf = moderate_conc_ppm * 1e-6
    if target_category == cd.CAT_SERIOUS:
        targ_conc_volf *= 10 # 10xERPG-3 for Serious
    for rel_dur_sec, pd_at_dur in phast_disp_dict.items():
        for wx, pd_at_dur_at_wx in pd_at_dur.items():
            for haz, pd_at_dur_at_wx_at_haz in pd_at_dur_at_wx.items():
                p_disp = pd_at_dur_at_wx_at_haz
                flattening = Flattening(conc_pfls=p_disp.conc_profiles)
                dist_m = flattening.calc_max_dist_at_conc(targ_conc_volf=targ_conc_volf, min_ht_m=0, max_ht_m=6)

    return dist_m

def model_run_for_solver(x0, args):
    target_input = args['target_input']
    row = args['row']
    m_io = get_m_io(row)
    target_category = args['target_category']
    adding_factor = 0
    if target_input == 'temp_deg_k':
        target_input = 'temp_deg_c'
        adding_factor = -273.15
    m_io.inputs[target_input] = x0 + adding_factor
    res = m_io.run()
    if res != ResultCode.SUCCESS:
        return None
    dist_m = parse_m_io(m_io, target_category=target_category)
    return dist_m

def get_sensitivity_around_init_values(row):
    output = []
    args = {
        'row' : row,
    }
    for target_category in targ_dists.keys():
        args['target_category'] = target_category
        for parameter_data in starting_inputs_list:
            args['parameter_data'] = parameter_data
            target_input = parameter_data['parameter']
            args['target_input'] = target_input
            increment_operation = parameter_data['increment_operation']
            increment_factor = parameter_data['increment_factor']
            x0 = parameter_data['initial_value']
            xhigh = increment_operation(x0, increment_factor)
            xlow = increment_operation(x0, increment_factor, inverse = True)
        
            for x in [xlow, x0, xhigh]:
                try:
                    dist_m = model_run_for_solver(x0=x, args=args)
                    output.append({
                        'chem_mix': row['chem_mix'],
                        'target_category': target_category,
                        'target_input' : target_input,
                        'input': x,
                        'value': dist_m,
                    })
                    print(output)
                except Exception as e:
                        print(f"{row['chem_mix']} could not be solved for {target_category} over {target_input} at {x}.  Error: {e.args} ")
                        continue
                
                try:
                    with open('sensitivity_output_acetaldehyde.csv', 'a', newline='') as csv_file:
                        writer = csv.DictWriter(f=csv_file, fieldnames=list(output[-1].keys()))
                        writer.writerows(output)
                        output = []
                except Exception as e:
                    print(f"Could not write results to file.  Will attempt to write next time.  \ncurrent results in buffer: \n{output}")
    return output

def solve_for_dists_at_concs(row):
    output = []
    args = {
        'row' : row,
    }
    for target_category in targ_dists.keys():
        target_distance = targ_dists[target_category]
        args['target_category'] = target_category
        for parameter_data in starting_inputs_list:
            args['parameter_data'] = parameter_data
            target_input = parameter_data['parameter']
            args['target_input'] = target_input
            x0 = parameter_data['initial_value']
            ll = parameter_data['lower_bound']
            ul = parameter_data['upper_bound']

            solver = Solver(arg_to_vary=x0, fxn_to_solve=model_run_for_solver, args=args, target=target_distance, ytol=1)
            solver.set_bisect_parameters(lower_limit=ll, upper_limit=ul, initial_value=x0)
            if solver.verify_bounds() is None or not solver.verify_bounds():
                continue
            try:
                if solver.solver_bisect():
                    output.append({
                        'chem_mix': row['chem_mix'],
                        'target_category': target_category,
                        'target_input' : target_input,
                        'value': solver.answer
                    })
            except Exception as e:
                print(f"{row['chem_mix']} could not be solved for {target_category} over {target_input}.  Error: {e.args} ")
                continue

    return output

def main():
    # output = []
    for _, row in test_cases_df.iterrows():
        get_sensitivity_around_init_values(row)
        # output.extend(chem_results)
        # if len(output) == 0:
        #     continue
        # try:
        #     with open('sensitivity_output.csv', 'a', newline='') as csv_file:
        #         writer = csv.DictWriter(f=csv_file, fieldnames=list(chem_results[-1].keys()))
        #         writer.writerows(output)
        #         output = []
        # except Exception as e:
        #     print(f"Could not write results to file.  Will attempt to write next time.  \ncurrent results in buffer: \n{output}")

if __name__ == '__main__':
    main()