import pickle
import pandas as pd

from pypws.materials import get_component_by_id

from py_lopa.calcs import helpers
from py_lopa.data.tables import Tables
from py_lopa.phast_io.phast_prep import prep_material


chem_cas_nos = helpers.get_dataframe_from_csv('data/chems.csv')
chem_cas_nos = chem_cas_nos['cas_no'].values.tolist()

cheminfo = helpers.get_dataframe_from_csv(Tables().CHEM_INFO)

mat_ids = [helpers.vlookup_value_x_in_pandas_dataframe_df_in_col_y_get_data_in_column_z(x=x, df = cheminfo, y="cas_no", z="mat_comp_id") for x in chem_cas_nos]
chem_names = [helpers.get_chem_name(x, cheminfo) for x in chem_cas_nos]

# get material component from pws

output = []
idx = -1
for cas_no, chem_name, mat_id in zip(chem_cas_nos, chem_names, mat_ids):
    idx += 1
    try:
        mat = prep_material(mat_ids=[mat_id], release_molfs=[1])
        data = mat.components[0].data_item
        nbp_deg_k = [d.equation_coefficients[0] for d in data if d.description == 'normalBoilingPoint'][0]
        
        output.append(
            {
                'cas_no': cas_no,
                'chem_name': chem_name,
                'mat_id': mat_id,
                'material': mat,
                'nbp_deg_k': nbp_deg_k
            }
        )
        with open('materials.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(output, f, pickle.DEFAULT_PROTOCOL)
        print(f'completed idx {idx} | {chem_name}')
    except Exception as e:
        print(f"could not save materials.  error msgs: {e.args}")




# build pure component material

# store pickle of dict:
# chem_name
# cas_no
# mat_id
# material
